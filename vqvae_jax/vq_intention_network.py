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

from brax.training.acme import running_statistics
from track_mjx.agent.observation_utils import normalizer_select


class VQEncoder(nn.Module):
    """VQ-VAE encoder that maps observations to continuous embeddings.

    When use_continuous_latent=False (default), outputs a single deterministic
    embedding z_e via a single Dense projection. When use_continuous_latent=True,
    outputs (z_e_discrete, cont_mean, cont_logvar) via three separate Dense
    projections — a discrete head for VQ quantization and a continuous head for
    VAE-style reparameterization with an independently sized latent.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latent_dim: Dimension of the discrete output embedding (must match codebook).
        continuous_latent_dim: Dimension of the continuous head (mean/logvar).
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
        use_continuous_latent: If True, output two-head (discrete + continuous).
    """

    layer_sizes: Sequence[int]
    latent_dim: int
    continuous_latent_dim: int = 4
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    use_continuous_latent: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        """Encode observations to continuous embedding.

        Args:
            x: Input observations, shape [..., input_dim].
            get_activation: If True, return intermediate activations.

        Returns:
            When use_continuous_latent=False:
                z_e: shape [..., latent_dim], or (z_e, activations) dict.
            When use_continuous_latent=True:
                (z_e_discrete, cont_mean, cont_logvar): shapes
                [..., latent_dim], [..., continuous_latent_dim],
                [..., continuous_latent_dim], or (..., activations) dict.
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

        if self.use_continuous_latent:
            # Discrete head: goes to VQ (reuses 'latent_projection' for compat)
            z_e_discrete = nn.Dense(self.latent_dim, name="latent_projection")(x)
            # Continuous head: separate dimension
            cont_mean = nn.Dense(self.continuous_latent_dim, name="continuous_mean")(x)
            cont_logvar = nn.Dense(
                self.continuous_latent_dim, name="continuous_logvar"
            )(x)
            if get_activation:
                activations["z_e_discrete"] = z_e_discrete
                activations["cont_mean"] = cont_mean
                activations["cont_logvar"] = cont_logvar
                return z_e_discrete, cont_mean, cont_logvar, activations
            return z_e_discrete, cont_mean, cont_logvar
        else:
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
    use_continuous_latent: bool = False
    continuous_latent_dim: int = 4
    use_ref_joints_encoder: bool = False

    def setup(self):
        """Initialize encoder, quantizer, and decoder submodules."""
        self.encoder = VQEncoder(
            layer_sizes=self.encoder_layers,
            latent_dim=self.latent_dim,
            continuous_latent_dim=self.continuous_latent_dim,
            use_continuous_latent=self.use_continuous_latent,
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
            key: JAX random key.
            deterministic: If True, skip noise and use mean for continuous latent.
            get_activation: If True, return intermediate activations.
            prev_indices: Optional indices from previous timestep for
                stickiness bias. Single array (depth=1 compat) or tuple.

        Returns:
            action_params: Shape [..., action_size*2].
            z_e: Discrete encoder output, shape [..., latent_dim].
                Used for VQ commitment loss.
            all_indices: Tuple of D index arrays, each shape [...].
            logvar: (cont_mean, cont_logvar) tuple when use_continuous_latent=True,
                else None. Each shape [..., continuous_latent_dim].
            (optional) extras: Dict of activations if get_activation=True
                (appended as 5th element).
        """
        traj = (
            obs["ref_joints"]
            if self.use_ref_joints_encoder
            else obs["task_obs"]
        )
        egocentric_obs = obs["proprioception"]

        # PRNG key management
        base_key = key[0] if key.ndim > 1 else key
        reparam_key = None
        if self.proprio_noise_scale > 0.0 and not deterministic:
            if self.use_continuous_latent and not deterministic:
                noise_key, reparam_key = jax.random.split(base_key)
            else:
                noise_key = base_key
            noise = (
                jax.random.normal(noise_key, egocentric_obs.shape)
                * self.proprio_noise_scale
            )
            egocentric_obs = egocentric_obs + noise
        elif self.use_continuous_latent and not deterministic:
            reparam_key = base_key

        if self.use_continuous_latent:
            # Two-head encoder: discrete + continuous
            if get_activation:
                z_e_discrete, cont_mean, cont_logvar, encoder_activations = (
                    self.encoder(traj, get_activation=True)
                )
            else:
                z_e_discrete, cont_mean, cont_logvar = self.encoder(traj)

            # Reparameterize continuous head
            if deterministic:
                z_e_sampled = cont_mean
            else:
                eps = jax.random.normal(reparam_key, cont_mean.shape)
                z_e_sampled = cont_mean + jnp.exp(0.5 * cont_logvar) * eps

            # VQ quantizes the discrete head
            z_hat_st, all_indices, all_z_q, all_residuals = self.quantizer(
                z_e_discrete, prev_indices=prev_indices
            )

            # Decoder input: [z_hat_st, z_e_sampled, proprioception]
            concatenated = jnp.concatenate(
                [z_hat_st, z_e_sampled, egocentric_obs], axis=-1
            )
            if get_activation:
                action, decoder_activations = self.decoder(
                    concatenated, get_activation=True
                )
                return (
                    action,
                    z_e_discrete,
                    all_indices,
                    (cont_mean, cont_logvar),
                    {
                        "encoder": encoder_activations,
                        "decoder": decoder_activations,
                        "egocentric_obs": egocentric_obs,
                        "traj_obs": traj,
                        "z_e_discrete": z_e_discrete,
                        "cont_mean": cont_mean,
                        "cont_logvar": cont_logvar,
                        "z_e_sampled": z_e_sampled,
                        "all_z_q": all_z_q,
                        "all_residuals": all_residuals,
                        "z_hat_st": z_hat_st,
                        "all_indices": all_indices,
                    },
                )
            else:
                action, _ = self.decoder(concatenated)
                return action, z_e_discrete, all_indices, (cont_mean, cont_logvar)
        else:
            # Original deterministic path
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
                    None,
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
                z_hat_st, all_indices, _, _ = self.quantizer(
                    z_e, prev_indices=prev_indices
                )
                concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
                action, _ = self.decoder(concatenated)
                return action, z_e, all_indices, None

    def forward_step_chunked(
        self,
        obs: Mapping[str, jnp.ndarray],
        held_d0_idx: jnp.ndarray,
        tau: jnp.ndarray,
        commitment_horizon: int,
        key: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        tuple[jnp.ndarray, ...],
        tuple[jnp.ndarray, jnp.ndarray] | None,
        tuple[jnp.ndarray, jnp.ndarray],
    ]:
        """Single-timestep forward with D0 temporal commitment.

        Args:
            obs: Dict obs, each value shape [B, ...] (single timestep).
            held_d0_idx: Currently held D0 code index, shape [B].
            tau: Timer value, shape [B]. 0 = manager step.
            commitment_horizon: H.
            key: PRNG key for noise/reparameterization.
            deterministic: If True, skip noise.

        Returns:
            action_params: Shape [B, action_size*2].
            z_e_or_mean: Shape [B, latent_dim].
            all_indices: D-tuple of index arrays, each shape [B].
            logvar: Shape [B, latent_dim] or None.
            new_chunk_state: Tuple of (new_held_d0, new_tau), each shape [B].
        """
        traj = (
            obs["ref_joints"]
            if self.use_ref_joints_encoder
            else obs["task_obs"]
        )
        egocentric_obs = obs["proprioception"]
        H = commitment_horizon

        # PRNG key management
        reparam_key = None
        if self.proprio_noise_scale > 0.0 and not deterministic and key is not None:
            if self.use_continuous_latent:
                noise_key, reparam_key = jax.random.split(key)
            else:
                noise_key = key
            noise = (
                jax.random.normal(noise_key, egocentric_obs.shape)
                * self.proprio_noise_scale
            )
            egocentric_obs = egocentric_obs + noise
        elif self.use_continuous_latent and not deterministic and key is not None:
            reparam_key = key

        if self.use_continuous_latent:
            z_e_discrete, cont_mean, cont_logvar = self.encoder(traj)
            if deterministic or reparam_key is None:
                z_e_sampled = cont_mean
            else:
                eps = jax.random.normal(reparam_key, cont_mean.shape)
                z_e_sampled = cont_mean + jnp.exp(0.5 * cont_logvar) * eps
            vq_input = z_e_discrete
        else:
            vq_input = self.encoder(traj)
            cont_mean = None
            cont_logvar = None
            z_e_sampled = None

        codebook_0 = self.quantizer.codebooks[0].embeddings
        has_d1 = self.rvq_depth >= 2

        # Manager vs worker
        is_manager = tau == 0  # [B]

        # Fresh D0 quantization (with stickiness toward held code)
        bias_d0 = self.quantizer._bias_per_level[0]
        fresh_d0_z_q, fresh_d0_idx, _ = _quantize_single_level(
            z_e=vq_input,
            codebook=codebook_0,
            num_codes=self.num_codes,
            latent_dim=self.latent_dim,
            stickiness_bias=bias_d0,
            prev_indices=held_d0_idx,
            use_rotation=False,
        )

        # Select fresh or held D0
        held_d0_z_q = codebook_0[held_d0_idx]
        d0_idx = jnp.where(is_manager, fresh_d0_idx, held_d0_idx)
        # Expand is_manager for broadcasting against [B, D] (handles scalar tau too)
        is_manager_expanded = jnp.expand_dims(is_manager, axis=-1)
        d0_z_q = jnp.where(is_manager_expanded, fresh_d0_z_q, held_d0_z_q)

        # STE for D0
        z_q_st_d0 = (
            vq_input - jax.lax.stop_gradient(vq_input) + jax.lax.stop_gradient(d0_z_q)
        )

        if has_d1:
            # D1: always fresh
            codebook_1 = self.quantizer.codebooks[1].embeddings
            residual = vq_input - jax.lax.stop_gradient(d0_z_q)
            _, d1_idx, z_q_st_d1 = _quantize_single_level(
                z_e=residual,
                codebook=codebook_1,
                num_codes=self.num_codes,
                latent_dim=self.latent_dim,
                stickiness_bias=0.0,
                prev_indices=None,
                use_rotation=self.quantizer.use_rotation,
            )
            z_hat_st = z_q_st_d0 + z_q_st_d1
        else:
            z_hat_st = z_q_st_d0
        if self.use_continuous_latent:
            concatenated = jnp.concatenate(
                [z_hat_st, z_e_sampled, egocentric_obs], axis=-1
            )
        else:
            concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
        action, _ = self.decoder(concatenated)

        # Update chunk state
        new_tau = (tau + 1) % H
        new_held_d0 = d0_idx

        logvar = (cont_mean, cont_logvar) if self.use_continuous_latent else None
        all_indices = (d0_idx, d1_idx) if has_d1 else (d0_idx,)
        return (
            action,
            vq_input,
            all_indices,
            logvar,
            (new_held_d0, new_tau),
        )

    def forward_temporal_chunked(
        self,
        obs: Mapping[str, jnp.ndarray],
        commitment_horizon: int,
        episode_mask: jnp.ndarray | None = None,
        key: jax.Array | None = None,
        initial_held_d0_idx: jnp.ndarray | None = None,
        initial_tau: jnp.ndarray | None = None,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        tuple[jnp.ndarray, ...],
        tuple[jnp.ndarray, jnp.ndarray] | None,
        jnp.ndarray,
    ]:
        """Forward pass with D0 temporal commitment (code chunking).

        D0 codes are held constant for commitment_horizon steps. When
        rvq_depth >= 2, D1 codes change freely every step.

        Args:
            obs: Dictionary observation with keys:
                - "imitation_target": Shape [T, B, traj_dim].
                - "proprioception": Shape [T, B, proprio_dim].
            commitment_horizon: H, number of steps to hold D0 code.
            episode_mask: Optional mask indicating episode boundaries.
                Shape [T, B]. Value of 0 indicates episode start.
            key: JAX random key for noise/reparameterization.
            initial_held_d0_idx: Optional initial D0 index, shape [B].
                Used to continue chunking from a previous unroll boundary.
            initial_tau: Optional initial tau, shape [B].
                Used to continue chunking from a previous unroll boundary.

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e_or_mean: Shape [T, B, latent_dim].
            all_indices: D-tuple of index arrays, each shape [T, B].
            logvar: Shape [T, B, latent_dim] when continuous=True, else None.
            tau: Timer values, shape [T, B].
        """
        traj = (
            obs["ref_joints"]
            if self.use_ref_joints_encoder
            else obs["task_obs"]
        )  # [T, B, traj_dim]
        egocentric_obs = obs["proprioception"]  # [T, B, proprio_dim]
        T, B = traj.shape[0], traj.shape[1]
        H = commitment_horizon

        # PRNG key management
        reparam_key = None
        if self.proprio_noise_scale > 0.0 and key is not None:
            if self.use_continuous_latent:
                noise_key, reparam_key = jax.random.split(key)
            else:
                noise_key = key
            noise = (
                jax.random.normal(noise_key, egocentric_obs.shape)
                * self.proprio_noise_scale
            )
            egocentric_obs = egocentric_obs + noise
        elif self.use_continuous_latent and key is not None:
            reparam_key = key

        if self.use_continuous_latent:
            z_e_discrete, cont_mean, cont_logvar = self.encoder(traj)
            if reparam_key is not None:
                eps = jax.random.normal(reparam_key, cont_mean.shape)
                z_e_sampled = cont_mean + jnp.exp(0.5 * cont_logvar) * eps
            else:
                z_e_sampled = cont_mean
            vq_input = z_e_discrete
        else:
            vq_input = self.encoder(traj)  # [T, B, latent_dim]
            cont_mean = None
            cont_logvar = None
            z_e_sampled = None

        # Get codebook(s)
        codebook_0 = self.quantizer.codebooks[0].embeddings
        has_d1 = self.rvq_depth >= 2

        # Prepare episode start mask: True = episode start
        if episode_mask is not None:
            is_episode_start = episode_mask == 0  # [T, B]
        else:
            is_episode_start = jnp.concatenate(
                [jnp.ones((1, B), dtype=bool), jnp.zeros((T - 1, B), dtype=bool)],
                axis=0,
            )

        # --- Step 1: Precompute tau via lightweight scan (carries [B] int32) ---
        if initial_tau is not None:
            init_tau = initial_tau
        else:
            init_tau = jnp.zeros((B,), dtype=jnp.int32)

        def tau_step(prev_tau, is_start_t):
            # At episode start, force manager step (tau=0) then advance to 1
            # Otherwise, advance the counter cyclically
            force_fresh = (prev_tau == 0) | is_start_t
            # cur_tau is the tau value *at* this step (before increment)
            cur_tau = jnp.where(is_start_t, jnp.zeros_like(prev_tau), prev_tau)
            # next_tau advances; episode start acts as manager so next is 1
            next_tau = jnp.where(
                is_start_t,
                jnp.ones_like(prev_tau) % H,
                (prev_tau + 1) % H,
            )
            return next_tau, (cur_tau, force_fresh)

        _, (tau, force_fresh) = jax.lax.scan(tau_step, init_tau, is_episode_start)
        # tau: [T, B], force_fresh: [T, B]

        # --- Step 2+3: D0 quantization with optional stickiness ---
        if initial_held_d0_idx is not None:
            init_d0_idx = initial_held_d0_idx
        else:
            init_d0_idx = jnp.zeros((B,), dtype=jnp.int32)

        bias_d0 = self.quantizer._bias_per_level[0]
        if bias_d0 <= 0:
            # Fast path: vectorized D0 quantization (no stickiness)
            _, fresh_d0_idx, _ = _quantize_single_level(
                z_e=vq_input,
                codebook=codebook_0,
                num_codes=self.num_codes,
                latent_dim=self.latent_dim,
                stickiness_bias=0.0,
                prev_indices=None,
                use_rotation=False,
            )

            def fill_fn(prev_idx, inputs_t):
                is_fresh_t, fresh_idx_t = inputs_t
                new_idx = jnp.where(is_fresh_t, fresh_idx_t, prev_idx)
                return new_idx, new_idx

            _, d0_indices = jax.lax.scan(
                fill_fn, init_d0_idx, (force_fresh, fresh_d0_idx)
            )
        else:
            # Sequential D0 quantization with stickiness toward held code
            def fill_with_stickiness_fn(prev_idx, inputs_t):
                is_fresh_t, vq_input_t = inputs_t  # vq_input_t: [B, D]
                _, fresh_idx_t, _ = _quantize_single_level(
                    z_e=vq_input_t,
                    codebook=codebook_0,
                    num_codes=self.num_codes,
                    latent_dim=self.latent_dim,
                    stickiness_bias=bias_d0,
                    prev_indices=prev_idx,
                    use_rotation=False,
                )
                new_idx = jnp.where(is_fresh_t, fresh_idx_t, prev_idx)
                return new_idx, new_idx

            _, d0_indices = jax.lax.scan(
                fill_with_stickiness_fn, init_d0_idx, (force_fresh, vq_input)
            )
        # d0_indices: [T, B]

        # --- Step 4: Look up D0 vectors and compute STE ---
        d0_z_q = codebook_0[d0_indices]  # [T, B, D]
        z_q_st_d0 = (
            vq_input - jax.lax.stop_gradient(vq_input) + jax.lax.stop_gradient(d0_z_q)
        )

        if has_d1:
            # --- Step 5: Vectorized D1 over all [T, B] at once ---
            codebook_1 = self.quantizer.codebooks[1].embeddings
            residual = vq_input - jax.lax.stop_gradient(d0_z_q)
            _, d1_indices, z_q_st_d1 = _quantize_single_level(
                z_e=residual,
                codebook=codebook_1,
                num_codes=self.num_codes,
                latent_dim=self.latent_dim,
                stickiness_bias=0.0,
                prev_indices=None,
                use_rotation=self.quantizer.use_rotation,
            )
            # --- Step 6: Combine ---
            z_hat_st = z_q_st_d0 + z_q_st_d1
        else:
            z_hat_st = z_q_st_d0

        # Decode all timesteps in parallel
        if self.use_continuous_latent:
            concatenated = jnp.concatenate(
                [z_hat_st, z_e_sampled, egocentric_obs], axis=-1
            )
        else:
            concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
        action, _ = self.decoder(concatenated)

        all_indices = (d0_indices, d1_indices) if has_d1 else (d0_indices,)
        logvar = (cont_mean, cont_logvar) if self.use_continuous_latent else None
        return action, vq_input, all_indices, logvar, tau

    def forward_temporal(
        self,
        obs: Mapping[str, jnp.ndarray],
        episode_mask: jnp.ndarray | None = None,
        key: jax.Array | None = None,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        tuple[jnp.ndarray, ...],
        tuple[jnp.ndarray, jnp.ndarray] | None,
    ]:
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
            key: JAX random key for noise/reparameterization.

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e_or_mean: Shape [T, B, latent_dim]. z_e when continuous=False,
                mean when continuous=True.
            all_indices: Tuple of D arrays, each shape [T, B].
            logvar: Shape [T, B, latent_dim] when continuous=True, else None.
        """
        traj = (
            obs["ref_joints"]
            if self.use_ref_joints_encoder
            else obs["task_obs"]
        )  # [T, B, traj_dim]
        egocentric_obs = obs["proprioception"]  # [T, B, proprio_dim]

        # PRNG key management for temporal path
        reparam_key = None
        if self.proprio_noise_scale > 0.0 and key is not None:
            if self.use_continuous_latent:
                noise_key, reparam_key = jax.random.split(key)
            else:
                noise_key = key
            noise = (
                jax.random.normal(noise_key, egocentric_obs.shape)
                * self.proprio_noise_scale
            )
            egocentric_obs = egocentric_obs + noise
        elif self.use_continuous_latent and key is not None:
            reparam_key = key

        if self.use_continuous_latent:
            # Two-head encoder: discrete + continuous
            z_e_discrete, cont_mean, cont_logvar = self.encoder(traj)

            # Reparameterize continuous head
            if reparam_key is not None:
                eps = jax.random.normal(reparam_key, cont_mean.shape)
                z_e_sampled = cont_mean + jnp.exp(0.5 * cont_logvar) * eps
            else:
                z_e_sampled = cont_mean

            # VQ quantizes the discrete head
            vq_input = z_e_discrete
        else:
            # Encode all timesteps in parallel
            vq_input = self.encoder(traj)  # [T, B, latent_dim]
            cont_mean = None
            cont_logvar = None
            z_e_sampled = None

        D = self.rvq_depth

        # If no bias, process in parallel (faster)
        if not self._has_stickiness:
            z_hat_st, all_indices, _, _ = self.quantizer(vq_input, prev_indices=None)
            if self.use_continuous_latent:
                concatenated = jnp.concatenate(
                    [z_hat_st, z_e_sampled, egocentric_obs], axis=-1
                )
            else:
                concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)
            logvar = (cont_mean, cont_logvar) if self.use_continuous_latent else None
            return action, vq_input, all_indices, logvar

        # Sequential quantization with stickiness bias using scan
        T, B = vq_input.shape[0], vq_input.shape[1]

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
        z_hat_st_0, indices_0, _, _ = self.quantizer(vq_input[0], prev_indices=None)

        if T > 1:
            init_carry = indices_0  # tuple of D arrays
            _, (z_hat_st_rest, indices_rest) = jax.lax.scan(
                quantize_step,
                init_carry,
                (vq_input[1:], scan_mask[1:]),
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
        if self.use_continuous_latent:
            concatenated = jnp.concatenate(
                [z_hat_st, z_e_sampled, egocentric_obs], axis=-1
            )
        else:
            concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
        action, _ = self.decoder(concatenated)

        logvar = (cont_mean, cont_logvar) if self.use_continuous_latent else None
        return action, vq_input, all_indices, logvar


class VQPolicyNetwork:
    """VQ-VAE policy network with standard, temporal, and chunked apply methods.

    This class wraps a VQIntentionNetwork and provides apply methods:
    - apply: Standard forward pass (parallel processing, optional prev_indices)
    - apply_temporal: Sequential processing with stickiness bias via scan
    - apply_temporal_chunked: D0 temporal commitment with free D1

    Attributes:
        init: Initialization function.
        apply: Standard apply function.
        apply_temporal: Temporal apply function with sequential quantization.
        apply_temporal_chunked: Chunked apply with D0 commitment.
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
        apply_temporal_chunked=None,
        apply_step_chunked=None,
    ):
        self.init = init
        self.apply = apply
        self.apply_temporal = apply_temporal
        self.apply_temporal_chunked = apply_temporal_chunked
        self.apply_step_chunked = apply_step_chunked
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
    use_continuous_latent: bool = False,
    continuous_latent_dim: int = 4,
    use_ref_joints_encoder: bool = False,
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
        use_ref_joints_encoder: If True, encoder reads obs["ref_joints"]
            instead of obs["task_obs"].

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
        use_continuous_latent=use_continuous_latent,
        continuous_latent_dim=continuous_latent_dim,
        use_ref_joints_encoder=use_ref_joints_encoder,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
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
            z_e_or_mean: Continuous encoder output or mean (for loss computation).
            all_indices: Tuple of D codebook index arrays.
            logvar: logvar when use_continuous_latent=True, else None.
            (optional) extras: Dict of activations if get_activation=True.
        """
        state_obs = obs["state"]
        ref_joints = state_obs.get("ref_joints")
        state_normalizer = normalizer_select(processor_params, "state")
        obs = dict(running_statistics.normalize(state_obs, state_normalizer))
        if ref_joints is not None:
            obs["ref_joints"] = ref_joints
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
            prev_indices=prev_indices,
        )

    def apply_temporal(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        episode_mask: jnp.ndarray | None = None,
        proprio_noise_key: jax.Array | None = None,
    ):
        """Apply VQ policy with temporal stickiness bias.

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e_or_mean: Shape [T, B, latent_dim].
            all_indices: Tuple of D arrays, each shape [T, B].
            logvar: Shape [T, B, latent_dim] or None.
        """
        state_obs = obs["state"]
        ref_joints = state_obs.get("ref_joints")
        state_normalizer = normalizer_select(processor_params, "state")
        obs = dict(running_statistics.normalize(state_obs, state_normalizer))
        if ref_joints is not None:
            obs["ref_joints"] = ref_joints
        return policy_module.apply(
            policy_params,
            obs=obs,
            episode_mask=episode_mask,
            key=proprio_noise_key,
            method=policy_module.forward_temporal,
        )

    def apply_temporal_chunked(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        commitment_horizon: int,
        episode_mask: jnp.ndarray | None = None,
        proprio_noise_key: jax.Array | None = None,
        initial_held_d0_idx: jnp.ndarray | None = None,
        initial_tau: jnp.ndarray | None = None,
    ):
        """Apply VQ policy with D0 temporal commitment (code chunking).

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e_or_mean: Shape [T, B, latent_dim].
            all_indices: D-tuple of index arrays, each [T, B].
            logvar: Shape [T, B, latent_dim] or None.
            tau: Timer values, shape [T, B].
        """
        state_obs = obs["state"]
        ref_joints = state_obs.get("ref_joints")
        state_normalizer = normalizer_select(processor_params, "state")
        obs = dict(running_statistics.normalize(state_obs, state_normalizer))
        if ref_joints is not None:
            obs["ref_joints"] = ref_joints
        return policy_module.apply(
            policy_params,
            obs=obs,
            commitment_horizon=commitment_horizon,
            episode_mask=episode_mask,
            key=proprio_noise_key,
            initial_held_d0_idx=initial_held_d0_idx,
            initial_tau=initial_tau,
            method=policy_module.forward_temporal_chunked,
        )

    def apply_step_chunked(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        held_d0_idx: jnp.ndarray,
        tau: jnp.ndarray,
        commitment_horizon: int,
        key: jax.Array | None = None,
        deterministic: bool = False,
    ):
        """Apply VQ policy for a single chunked timestep.

        Returns:
            action_params: Shape [B, action_size*2].
            z_e_or_mean: Shape [B, latent_dim].
            all_indices: D-tuple of index arrays, each [B].
            logvar: Shape [B, latent_dim] or None.
            new_chunk_state: Tuple of (new_held_d0, new_tau), each [B].
        """
        state_obs = obs["state"]
        ref_joints = state_obs.get("ref_joints")
        state_normalizer = normalizer_select(processor_params, "state")
        obs = dict(running_statistics.normalize(state_obs, state_normalizer))
        if ref_joints is not None:
            obs["ref_joints"] = ref_joints
        return policy_module.apply(
            policy_params,
            obs=obs,
            held_d0_idx=held_d0_idx,
            tau=tau,
            commitment_horizon=commitment_horizon,
            key=key,
            deterministic=deterministic,
            method=policy_module.forward_step_chunked,
        )

    # Create dummy dict observation for initialization
    dummy_obs = {
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    if use_ref_joints_encoder and "ref_joints" in obs_sizes:
        dummy_obs["ref_joints"] = jnp.zeros((1, obs_sizes["ref_joints"]))
    dummy_key = jax.random.PRNGKey(0)

    return VQPolicyNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
        apply_temporal=apply_temporal,
        apply_temporal_chunked=apply_temporal_chunked,
        apply_step_chunked=apply_step_chunked,
        stickiness_bias=stickiness_bias,
        num_codes=num_codes,
        latent_dim=latent_dim,
        rvq_depth=rvq_depth,
    )
