"""Encoder-decoder policy network for KPMS-driven motor control.

The network receives a pre-computed syllable code (integer, passed as float in
``obs["kpms_code"]``), the agent's proprioception, and optionally the reference
trajectory (``obs["imitation_target"]``). It embeds the code via a learned
embedding table, optionally encodes the reference trajectory into a continuous
latent, and feeds the concatenation through an MLP decoder to produce action
parameters.

When ``use_continuous_encoder=False`` (default), the continuous encoder is
skipped and behavior is identical to the original decoder-only network.
"""

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MoSeqEncoderDecoderNetwork(nn.Module):
    """Encoder-decoder policy: code embedding + continuous latent + proprio -> action.

    Attributes:
        num_codes: Number of syllable codes (embedding table rows).
        code_embed_dim: Dimensionality of the code embedding.
        decoder_layer_sizes: Hidden layer sizes for the decoder MLP.
        action_param_size: Output dimension (2 * action_dim for NormalTanh).
        activation: Activation function.
        kernel_init: Initializer for Dense layers.
        use_continuous_encoder: Whether to encode imitation_target into z_e.
        encoder_layer_sizes: Hidden layer sizes for the encoder MLP.
        continuous_latent_dim: Dimensionality of the continuous latent (= code_embed_dim).
    """

    num_codes: int = 32
    code_embed_dim: int = 16
    decoder_layer_sizes: Sequence[int] = (512, 512, 256, 256)
    action_param_size: int = 1
    activation: Callable = nn.silu
    kernel_init: Callable = nn.initializers.lecun_uniform()
    use_continuous_encoder: bool = False
    encoder_layer_sizes: Sequence[int] = (256, 128)
    continuous_latent_dim: int = 16

    def setup(self):
        self.code_embedding = nn.Embed(
            num_embeddings=self.num_codes,
            features=self.code_embed_dim,
        )

    @nn.compact
    def __call__(
        self,
        obs: dict[str, jnp.ndarray],
        key=None,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Forward pass.

        Args:
            obs: Observation dict with ``kpms_code`` (raw int-as-float,
                shape ``[..., 1]``), ``proprioception`` (normalized,
                shape ``[..., proprio_dim]``), and optionally
                ``imitation_target`` (normalized, shape ``[..., target_dim]``).
            key: PRNG key for reparameterization (used when
                ``use_continuous_encoder=True`` and ``deterministic=False``).
            deterministic: If True, use mean (no sampling) for continuous latent.

        Returns:
            Tuple of ``(action_params, code_idx, mean, logvar)``.
            When ``use_continuous_encoder=False``, ``mean`` and ``logvar``
            are ``None``.
        """
        # Extract code index — passed as float, round to int for embedding
        kpms_code = obs["kpms_code"]  # [..., 1]
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)
        code_emb = self.code_embedding(code_idx)  # [..., code_embed_dim]

        # Proprioception (already normalized and flattened by the policy wrapper)
        proprio = obs["proprioception"]

        # Continuous encoder (optional)
        if self.use_continuous_encoder:
            imitation_target = obs["imitation_target"]

            # Encoder MLP
            h = imitation_target
            for i, size in enumerate(self.encoder_layer_sizes):
                h = nn.Dense(size, kernel_init=self.kernel_init, name=f"enc_{i}")(h)
                h = self.activation(h)
                h = nn.LayerNorm(name=f"enc_ln_{i}")(h)

            mean = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="continuous_mean",
            )(h)
            logvar = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="continuous_logvar",
            )(h)

            # Reparameterization
            if deterministic:
                z_e = mean
            else:
                if key is None:
                    key = self.make_rng("params")
                eps = jax.random.normal(key, mean.shape)
                z_e = mean + jnp.exp(0.5 * logvar) * eps

            # Decoder input: code_emb + z_e + proprio
            x = jnp.concatenate([code_emb, z_e, proprio], axis=-1)
        else:
            mean = None
            logvar = None
            # Decoder input: code_emb + proprio (original behavior)
            x = jnp.concatenate([code_emb, proprio], axis=-1)

        # Decoder MLP
        for i, size in enumerate(self.decoder_layer_sizes):
            x = nn.Dense(size, kernel_init=self.kernel_init, name=f"dec_{i}")(x)
            x = self.activation(x)
            x = nn.LayerNorm(name=f"dec_ln_{i}")(x)
        action_params = nn.Dense(
            self.action_param_size,
            kernel_init=self.kernel_init,
            name="action_head",
        )(x)

        return action_params, code_idx, mean, logvar


# Backward-compatible alias
MoSeqDecoderNetwork = MoSeqEncoderDecoderNetwork
