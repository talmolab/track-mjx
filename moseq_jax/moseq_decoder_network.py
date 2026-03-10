"""Decoder-only policy network for KPMS-driven motor control.

The network receives a pre-computed syllable code (integer, passed as float in
``obs["kpms_code"]``) and the agent's proprioception. It embeds the code via a
learned embedding table and feeds the concatenation through an MLP decoder to
produce action parameters.

No encoder or vector quantizer is involved — codes are externally supplied by
Keypoint-MoSeq.
"""

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MoSeqDecoderNetwork(nn.Module):
    """Decoder-only policy: code embedding + proprioception -> action params.

    Attributes:
        num_codes: Number of syllable codes (embedding table rows).
        code_embed_dim: Dimensionality of the code embedding.
        decoder_layer_sizes: Hidden layer sizes for the decoder MLP.
        action_param_size: Output dimension (2 * action_dim for NormalTanh).
        activation: Activation function.
        kernel_init: Initializer for Dense layers.
    """

    num_codes: int = 32
    code_embed_dim: int = 16
    decoder_layer_sizes: Sequence[int] = (512, 512, 256, 256)
    action_param_size: int = 1
    activation: Callable = nn.silu
    kernel_init: Callable = nn.initializers.lecun_uniform()

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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: Observation dict with ``kpms_code`` (raw int-as-float,
                shape ``[..., 1]``) and ``proprioception`` (normalized,
                shape ``[..., proprio_dim]``).
            key: Unused PRNG key (kept for interface compatibility).
            deterministic: Unused (kept for interface compatibility).

        Returns:
            Tuple of ``(action_params, code_idx)``.
        """
        # Extract code index — passed as float, round to int for embedding
        kpms_code = obs["kpms_code"]  # [..., 1]
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)
        code_emb = self.code_embedding(code_idx)  # [..., code_embed_dim]

        # Proprioception (already normalized and flattened by the policy wrapper)
        proprio = obs["proprioception"]

        # Decoder MLP
        x = jnp.concatenate([code_emb, proprio], axis=-1)
        for i, size in enumerate(self.decoder_layer_sizes):
            x = nn.Dense(size, kernel_init=self.kernel_init, name=f"dec_{i}")(x)
            x = self.activation(x)
            x = nn.LayerNorm(name=f"dec_ln_{i}")(x)
        action_params = nn.Dense(
            self.action_param_size,
            kernel_init=self.kernel_init,
            name="action_head",
        )(x)

        return action_params, code_idx
