"""Motion predictor: z_t -> next N motion frames."""
from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionPredictor(nn.Module):
    """f(z_t) = S_hat^motion_{t+1} (a window of length `horizon`)."""

    layer_sizes: Sequence[int] = (256, 128)
    horizon: int = 5
    feat_dim: int = 77

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(z)
        flat_out = nn.Dense(self.horizon * self.feat_dim, name="pred_head")(h)
        return flat_out.reshape(batch, self.horizon, self.feat_dim)


class MotionPredictorSplit(nn.Module):
    """Body-part-split predictor: each branch's z slice predicts its own feats.

    The total latent z = concat(z_branch_0, z_branch_1, ...). For each branch,
    a separate `MotionPredictor` maps z_branch -> the branch's slice of the
    full feat vector at horizon steps in the future. Outputs are scattered
    back into a (B, horizon, full_feat_dim) tensor by `branch_output_indices`,
    so the result has the same shape as `MotionPredictor`'s output and slots
    into the same downstream code (encode_normalized → r_mimic).
    """

    branch_output_indices: Tuple[Tuple[int, ...], ...]
    branch_latent_dims: Tuple[int, ...]
    full_feat_dim: int
    horizon: int = 5
    layer_sizes: Sequence[int] = (256, 128)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        out = jnp.zeros((batch, self.horizon, self.full_feat_dim), dtype=z.dtype)
        offset = 0
        for i, (idx, ld) in enumerate(
            zip(self.branch_output_indices, self.branch_latent_dims)
        ):
            z_branch = jax_lax_dynamic_slice(z, offset, int(ld))
            offset += int(ld)
            pred_branch = MotionPredictor(
                layer_sizes=tuple(self.layer_sizes),
                horizon=self.horizon,
                feat_dim=len(idx),
                name=f"branch_{i}",
            )(z_branch)
            idx_arr = jnp.asarray(idx, dtype=jnp.int32)
            out = out.at[:, :, idx_arr].set(pred_branch)
        return out


def jax_lax_dynamic_slice(z: jnp.ndarray, offset: int, length: int) -> jnp.ndarray:
    # Plain Python slice — offset/length are concrete Python ints (Flax
    # iteration over module config). Avoids static-shape issues.
    return z[:, offset:offset + length]
