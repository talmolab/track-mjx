"""Motion encoder: window of m_t -> (mean, logvar) of latent z."""
from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionEncoder(nn.Module):
    """Flat-MLP encoder q(z | S^motion_t) returning (mean, logvar)."""

    layer_sizes: Sequence[int] = (256, 128)
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim)
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(x)
        mean = nn.Dense(self.latent_dim, name="mean_head")(h)
        logvar = nn.Dense(self.latent_dim, name="logvar_head")(h)
        return mean, logvar


class MotionEncoderConv1D(nn.Module):
    """Temporal-conv encoder q(z | S^motion_t) returning (mean, logvar).

    Treats the input window as a (time, feat) sequence and applies stacked
    1D convolutions with translation invariance along the time axis. Far
    more parameter-efficient than flattening the window into a flat MLP, and
    gives the encoder an inductive bias for motion patterns (it can learn to
    extract velocity from positional deltas across adjacent frames).
    """

    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim) — flax Conv interprets last axis as channels.
        h = x
        for i, ch in enumerate(self.conv_channels):
            h = nn.Conv(
                features=ch,
                kernel_size=(self.kernel_size,),
                padding="VALID",
                name=f"conv_{i}",
            )(h)
            h = nn.elu(h)
        h = h.reshape(h.shape[0], -1)
        if self.head_layer_sizes:
            h = Mlp(
                layer_sizes=tuple(self.head_layer_sizes),
                activate_final=True,
                name="head",
            )(h)
        mean = nn.Dense(self.latent_dim, name="mean_head")(h)
        logvar = nn.Dense(self.latent_dim, name="logvar_head")(h)
        return mean, logvar


class MotionEncoderConv1DSplit(nn.Module):
    """Body-part-split temporal-conv encoder.

    Runs `MotionEncoderConv1D` independently on slices of the input feature
    vector — one per body group — and concatenates the resulting (mean,
    logvar) outputs. This guarantees that each group has dedicated latent
    capacity instead of competing with high-variance groups (e.g. torso/head)
    in a shared bottleneck.

    `branch_input_indices` is a tuple of int-tuples giving, for each branch,
    the feat-dim indices the branch consumes. `branch_latent_dims` gives the
    per-branch latent dimensionality. The output latent dim is the sum.
    """

    branch_input_indices: Tuple[Tuple[int, ...], ...]
    branch_latent_dims: Tuple[int, ...]
    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        means, lvs = [], []
        for i, (idx, ld) in enumerate(
            zip(self.branch_input_indices, self.branch_latent_dims)
        ):
            x_branch = x[..., jnp.asarray(idx, dtype=jnp.int32)]
            mu_b, lv_b = MotionEncoderConv1D(
                conv_channels=tuple(self.conv_channels),
                kernel_size=self.kernel_size,
                head_layer_sizes=tuple(self.head_layer_sizes),
                latent_dim=int(ld),
                name=f"branch_{i}",
            )(x_branch)
            means.append(mu_b)
            lvs.append(lv_b)
        return jnp.concatenate(means, axis=-1), jnp.concatenate(lvs, axis=-1)


class MotionEncoderConv1DSharedSplit(nn.Module):
    """Shared-backbone, branched-output encoder.

    Mid-point in design space between `MotionEncoderConv1D` (single backbone,
    single z) and `MotionEncoderConv1DSplit` (independent per-branch backbones).
    A single conv1d stack sees ALL input features so cross-branch correlations
    (e.g. gait-phase ↔ root velocity, or limb-stride-phase ↔ torso pitch) are
    available in the hidden representation. Then the final layer branches into
    per-branch (μ, logvar) heads producing the per-branch latent slices that
    Phase 2's per-branch r_mimic still consumes.

    Compared to `MotionEncoderConv1DSplit`:
      - INPUT is the full feat vector (not sliced per branch).
      - Conv stack is SHARED — single set of weights, sees all features at
        every layer. This is the coupling point.
      - Final dense heads are split per branch (one Dense for μ, one for
        logvar, per branch). Concatenated output preserves the same downstream
        ordering as the non-shared split — so v9c-style branch_indices.npz
        and per-branch KL machinery in env_wrapper Just Works.

    Tested hypothesis: v13/v14's encoder disjoint kept root encoder from ever
    seeing leg phase, so z couldn't represent gait-phase-coupled motion as a
    single direction. Phase 2 root tracking suffered (6.7× worse per-step root
    drift than the kinematic baseline). A shared backbone should restore that
    coupling while keeping the per-branch latent allocation that the
    differentially-weighted r_mimic exploits.
    """

    branch_latent_dims: Tuple[int, ...]
    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim) — ALL features (no per-branch slicing).
        h = x
        for i, ch in enumerate(self.conv_channels):
            h = nn.Conv(
                features=ch,
                kernel_size=(self.kernel_size,),
                padding="VALID",
                name=f"conv_{i}",
            )(h)
            h = nn.elu(h)
        h = h.reshape(h.shape[0], -1)
        if self.head_layer_sizes:
            h = Mlp(
                layer_sizes=tuple(self.head_layer_sizes),
                activate_final=True,
                name="shared_head",
            )(h)
        # Branched (μ, logvar) projections — these are the only point where the
        # per-branch identity matters. Each branch gets its own Dense pair.
        means, lvs = [], []
        for i, ld in enumerate(self.branch_latent_dims):
            mu_b = nn.Dense(int(ld), name=f"mean_head_{i}")(h)
            lv_b = nn.Dense(int(ld), name=f"logvar_head_{i}")(h)
            means.append(mu_b)
            lvs.append(lv_b)
        return jnp.concatenate(means, axis=-1), jnp.concatenate(lvs, axis=-1)
