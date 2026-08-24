"""Recurrent (GRU) from-scratch vision DMPO networks — no prior, no decoder.

This is the control arm for the kl-anchor recurrent networks in
networks_kl_anchor_rnn.py. Same binocular CNN torso, same stacked-GRU memory,
same critic — but the policy emits actions **directly** instead of steering a
frozen imitation decoder through a bounded latent residual.

What is deliberately absent, relative to `_BAggressiveRecurrentPolicyNet`:

  * **Prior and Decoder modules.** There is no imitation checkpoint to splice,
    so `_splice_warm_start` is not called and `policy.init` is left alone.
  * **The residual transform** (`sigma_ball` / `sigma_tanh` / `tanh`). Those
    modes bound the policy's chosen reparameterization noise to the range the
    frozen decoder was trained on. With no decoder there is no training domain
    to stay inside, and the action bound is applied at the env boundary by
    `action_utils.bind` exactly as on the other from-scratch path.
  * **The zero-init `residual` Dense.** Its purpose was to make the step-0
    policy reproduce the frozen prior->decoder pipeline bit-for-bit
    (`r_anchor == 1.0000`). There is no warm start to preserve here, so the
    action head uses `GaussianPolicyHead`'s standard small-but-nonzero init —
    zero-initialising the mean would start every episode with an identically
    zero action, which is a worse exploration start, not a safer one.

The hidden-state contract is unchanged and is what the learner keys on: a
tuple of per-layer carry arrays, `[H_l]` unbatched / `[B, H_l]` batched, GRU
only. The learner's BPTT path is generic over this — it requires only
`policy.apply(params, obs, hidden, method="raw") -> (mu, scale, new_hidden)`
returning plain arrays (tfd distributions cannot be `lax.scan` carries), which
is why this module can reuse it without touching learner.py.

The critic is `_ValueVisionCriticNet`, the SAME class and shape the kl-anchor
recurrent arm uses, so a from-scratch arm and a prior-grafted arm differ in
the policy only. It stays feed-forward on purpose (matches the recurrent-PPO
precedent; Q's memory limitation is a documented v1 tradeoff).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks import DMPONetworks, GaussianPolicyHead
from track_mjx.agent.dmpo.networks_kl_anchor_rnn import RecurrentPolicyMeta
from track_mjx.agent.dmpo.networks_vision_bottleneck import _ValueVisionCriticNet
from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder
from track_mjx.agent.recurrent_ppo.networks import get_rnn_cell

tfd = tfp.distributions


class _ScratchRecurrentPolicyHeadModule(nn.Module):
    """CNN + task_obs + proprio -> MLP -> stacked GRUs -> action Gaussian.

    Layer naming (`hidden_i` / `LayerNorm_i`) matches
    `_RecurrentPolicyHeadModule` so param trees read the same way across the
    two arms; only the terminal head differs (action Gaussian here, zero-init
    latent residual there).

    The action head lives INSIDE this module on purpose. `label_param_tree`
    buckets by top-level block name under `params["params"]`, and anything it
    does not recognise falls into "other" — a separate `action_head` block
    would silently be driven by `other_lr_mult` (which no config here sets)
    instead of `policy_head_lr_mult`. Keeping it here makes `policy_head` the
    single trainable block, exactly as on the grafted arm.
    """

    action_size: int
    mlp_layers: Sequence[int]
    hidden_sizes: Sequence[int]
    vision_shape: tuple
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    cell_type: str = "gru"

    @nn.compact
    def __call__(self, vision, task_obs, proprio, hidden):
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(vision)
        h = jnp.concatenate([vis, task_obs, proprio], axis=-1)
        for i, size in enumerate(self.mlp_layers):
            h = nn.Dense(size, name=f"hidden_{i}")(h)
            h = nn.LayerNorm(name=f"LayerNorm_{i}")(h)
            h = nn.silu(h)
        new_hidden = []
        for layer, size in enumerate(self.hidden_sizes):
            cell = get_rnn_cell(self.cell_type, size)
            carry, h = cell(hidden[layer], h)
            new_hidden.append(carry)
        dist = GaussianPolicyHead(action_size=self.action_size, name="action")(h)
        # .loc / .stddev() are plain arrays; MultivariateNormalDiag's stddev is
        # exactly the scale_diag it was built with. Returning arrays (not the
        # distribution) is what makes this scan-safe in the BPTT unroll.
        return dist.loc, dist.stddev(), tuple(new_hidden)


class _ScratchRecurrentPolicyNet(nn.Module):
    """Recurrent end-to-end policy: obs + hidden -> action Gaussian.

    Two apply paths, mirroring `_BAggressiveRecurrentPolicyNet`:
      - `__call__(obs, hidden) -> (tfd.MultivariateNormalDiag, new_hidden)`
        for acting;
      - `raw(obs, hidden) -> (mu, scale, new_hidden)` for the learner's BPTT
        `lax.scan`, which cannot carry tfd objects.

    The module keeps the submodule name `policy_head` so
    `optim_kl_anchor.label_param_tree` labels it exactly as it labels the
    kl-anchor arms' trainable block — on this arm that is the whole network,
    which is the point, and it means `kl_anchor.policy_head_lr_mult` drives
    every parameter here rather than only some of them.
    """

    action_size: int
    vision_shape: tuple
    rnn_mlp_layers: Sequence[int]
    rnn_hidden_sizes: Sequence[int]
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    rnn_cell: str = "gru"

    def setup(self):
        self.policy_head = _ScratchRecurrentPolicyHeadModule(
            action_size=self.action_size,
            mlp_layers=tuple(self.rnn_mlp_layers),
            hidden_sizes=tuple(self.rnn_hidden_sizes),
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            cell_type=self.rnn_cell,
            name="policy_head",
        )

    def raw(self, obs, hidden):
        return self.policy_head(
            obs["vision"], obs["imitation_target"], obs["proprioception"], hidden
        )

    def __call__(self, obs, hidden):
        mu, scale, new_hidden = self.raw(obs, hidden)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale), new_hidden


def make_dmpo_scratch_rnn_networks(
    *,
    proprio_size: int,
    task_obs_size: int,
    action_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    rnn_cell: str = "gru",
    rnn_mlp_layers: Sequence[int] = (512,),
    rnn_hidden_sizes: Sequence[int] = (512,),
    rnn_store_dtype: Any = "float16",
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    critic_use_proprio: bool = False,
) -> DMPONetworks:
    """Build recurrent from-scratch DMPO policy + FF critic.

    Signature intentionally parallels `make_dmpo_kl_anchor_rnn_networks` minus
    everything prior/decoder-shaped (`latent_size`, `prior_layer_sizes`,
    `decoder_layer_sizes`, `warm_start_*`, `residual_mode`, `residual_scale`),
    so a config can be moved between the two arms by deleting keys rather than
    renaming them.

    Args:
        proprio_size / task_obs_size: accepted for signature parity and
            logging; the networks read their widths from the obs dict at init.
        action_size: action dimensionality (38 for this rodent) — emitted
            directly here rather than decoded from a latent.
        rnn_mlp_layers: pre-RNN Dense+LayerNorm+silu mixing stage.
        rnn_hidden_sizes: stacked GRU cell widths.
        rnn_store_dtype: replay dtype for the stored per-step hidden; the live
            hidden always runs float32. Accepts a dtype or its string name
            (hydra configs pass strings).

    Returns:
        `DMPONetworks(policy, critic, recurrent_meta)` — `recurrent_meta` is
        the trace-time switch rollout/learner/eval branch on.
    """
    del proprio_size, task_obs_size  # read from the obs dict at init.

    if rnn_cell != "gru":
        raise ValueError(
            f"policy_head_rnn cell {rnn_cell!r} is not supported in v1; "
            "only 'gru' (the tuple-of-arrays hidden contract has no room "
            "for LSTM (c, h) pairs)."
        )

    policy = _ScratchRecurrentPolicyNet(
        action_size=action_size,
        vision_shape=vision_shape,
        rnn_mlp_layers=tuple(rnn_mlp_layers),
        rnn_hidden_sizes=tuple(rnn_hidden_sizes),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        rnn_cell=rnn_cell,
    )

    critic = _ValueVisionCriticNet(
        layer_sizes=tuple(value_hidden_layer_sizes),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        vision_shape=vision_shape,
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        use_proprio=critic_use_proprio,
    )

    meta = RecurrentPolicyMeta(
        cell_type=rnn_cell,
        hidden_sizes=tuple(rnn_hidden_sizes),
        store_dtype=jnp.dtype(rnn_store_dtype),
    )

    return DMPONetworks(policy=policy, critic=critic, recurrent_meta=meta)
