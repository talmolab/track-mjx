"""Motion decoder used only for autoencoder pre-training; discarded for RL."""
from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionDecoder(nn.Module):
    """p(S^motion_t | z) reconstructing the full input window."""

    layer_sizes: Sequence[int] = (128, 256)
    window_len: int = 10
    feat_dim: int = 77

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(z)
        flat_out = nn.Dense(self.window_len * self.feat_dim, name="recon_head")(h)
        return flat_out.reshape(batch, self.window_len, self.feat_dim)


class MotionDecoderSplit(nn.Module):
    """Body-part-split decoder: each branch reconstructs its own slice of the
    feat vector. Outputs are scattered into a (B, window_len, full_feat_dim)
    tensor matching `MotionDecoder`'s output shape, so the same recon-MSE loss
    works without changes."""

    branch_output_indices: Tuple[Tuple[int, ...], ...]
    branch_latent_dims: Tuple[int, ...]
    full_feat_dim: int
    window_len: int = 10
    layer_sizes: Sequence[int] = (128, 256)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        out = jnp.zeros((batch, self.window_len, self.full_feat_dim), dtype=z.dtype)
        offset = 0
        for i, (idx, ld) in enumerate(
            zip(self.branch_output_indices, self.branch_latent_dims)
        ):
            z_branch = z[:, offset:offset + int(ld)]
            offset += int(ld)
            dec_branch = MotionDecoder(
                layer_sizes=tuple(self.layer_sizes),
                window_len=self.window_len,
                feat_dim=len(idx),
                name=f"branch_{i}",
            )(z_branch)
            idx_arr = jnp.asarray(idx, dtype=jnp.int32)
            out = out.at[:, :, idx_arr].set(dec_branch)
        return out
