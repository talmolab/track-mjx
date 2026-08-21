"""Recurrent (GRU) variant of the B-aggressive kl-anchor DMPO networks.

Same Prior + Decoder + PolicyHead decomposition as networks_kl_anchor.py,
but the policy head carries a stacked-GRU state between steps so the policy
can *remember* (e.g. gap edges that have scrolled out of view) — the memory
channel the pure-sparse gap-jump arms need. Prior and Decoder are the SAME
modules, imported from networks_kl_anchor (not copied), so the imitation
checkpoint splice in `_splice_warm_start` works verbatim, and the head keeps
the module name `policy_head` so `optim_kl_anchor.label_param_tree` labels
it without changes.

The step-0 warm-start invariant survives recurrence: the head's last Dense
("residual") is zero-initialised, so the residual is exactly 0 for ANY
hidden state (including arbitrary garbage) and the policy reproduces the
frozen prior->decoder pipeline bit-for-bit at initialisation — the GRU only
changes what the head *can* express after training, not what it expresses
at step 0.

Hidden-state convention (the pinned cross-module contract): a **tuple of
per-layer carry arrays**, shapes `[H_l]` unbatched / `[B, H_l]` batched.
v1 is GRU-only, so each carry is a single array (no LSTM (c, h) pairs).
Modules are shape-agnostic — the codebase convention is `jax.vmap` of an
unbatched apply over envs, so per-env hidden slices are `[H_l]`.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _DecoderModule,
    _PriorModule,
    _splice_warm_start,
)
from track_mjx.agent.dmpo.networks_vision_bottleneck import (
    _ValueVisionCriticNet,
)
from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder
from track_mjx.agent.recurrent_ppo.networks import get_rnn_cell

tfd = tfp.distributions


class RecurrentPolicyMeta(NamedTuple):
    """Static description of the recurrent policy's hidden state.

    Threaded through `DMPONetworks.recurrent_meta` so rollout / learner /
    eval can build and store hidden states without importing this module's
    network classes (they only need shapes + storage dtype). `store_dtype`
    is the replay-buffer dtype for the per-step hidden (f16 halves the
    transition growth); the live hidden always runs in float32.
    """

    cell_type: str
    hidden_sizes: tuple
    store_dtype: Any

    def init_hidden(self, batch_size: Optional[int] = None) -> tuple:
        """Zero hidden state: `([B, H_l], ...)` or `([H_l], ...)` if unbatched."""
        if batch_size is None:
            return tuple(jnp.zeros((h,)) for h in self.hidden_sizes)
        return tuple(jnp.zeros((batch_size, h)) for h in self.hidden_sizes)


class _RecurrentPolicyHeadModule(nn.Module):
    """CNN + task_obs + proprio -> MLP -> stacked GRUs -> residual.

    Identical to the feed-forward `_PolicyHeadModule` up to the MLP mixing
    stage (same `hidden_i` / `LayerNorm_i` naming), then routes through one
    GRU cell per `hidden_sizes` entry before the zero-init `residual` Dense.
    Zero-init keeps residual == 0 at step 0 regardless of the hidden state,
    which is what preserves the warm-start invariant (r_anchor = 1.0).
    """

    mlp_layers: Sequence[int]
    hidden_sizes: Sequence[int]
    latent_size: int
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
        residual = nn.Dense(
            self.latent_size,
            name="residual",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        return residual, tuple(new_hidden)


class _BAggressiveRecurrentPolicyNet(nn.Module):
    """Trainable Prior + recurrent PolicyHead + Decoder.

    Two apply paths:
      - `__call__(obs, hidden) -> (tfd.MultivariateNormalDiag, new_hidden)`
        for acting (sample/mode on the dist);
      - `raw(obs, hidden) -> (mu, scale, new_hidden)` via
        `policy.apply(params, obs, hidden, method="raw")` — plain arrays so
        the learner's BPTT `lax.scan` can stack per-step outputs (tfd
        distributions are not scan carries/outputs).

    The residual transform matches `_BAggressivePolicyNet` exactly (see
    networks_kl_anchor.py for the full measured rationale behind each mode);
    only the head is recurrent.
    """

    action_size: int
    latent_size: int
    vision_shape: tuple
    prior_layer_sizes: Sequence[int]
    decoder_layer_sizes: Sequence[int]
    rnn_mlp_layers: Sequence[int]
    rnn_hidden_sizes: Sequence[int]
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    rnn_cell: str = "gru"
    # sigma_ball is the default here (unlike the FF net's sigma_tanh): this
    # module postdates the measured per-dim-box exploit (see the sigma_ball
    # comment below) and every live sigma-mode arm on this branch runs
    # sigma_ball. residual_scale is the ball radius r (chi^2_16 95th pct
    # ~5.13; arms run 5.0).
    residual_mode: str = "sigma_ball"
    residual_scale: float = 5.0

    def setup(self):
        self.prior = _PriorModule(
            layer_sizes=tuple(self.prior_layer_sizes),
            latents=self.latent_size,
            name="prior",
        )
        self.decoder = _DecoderModule(
            layer_sizes=tuple(self.decoder_layer_sizes) + (2 * self.action_size,),
            name="decoder",
        )
        self.policy_head = _RecurrentPolicyHeadModule(
            mlp_layers=tuple(self.rnn_mlp_layers),
            hidden_sizes=tuple(self.rnn_hidden_sizes),
            latent_size=self.latent_size,
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            cell_type=self.rnn_cell,
            name="policy_head",
        )

    def raw(self, obs, hidden):
        vision = obs["vision"]
        task_obs = obs["imitation_target"]
        proprio = obs["proprioception"]

        z_prior, z_logvar = self.prior(proprio)
        raw, new_hidden = self.policy_head(vision, task_obs, proprio, hidden)

        # Residual transform copied from _BAggressivePolicyNet — the residual
        # is the policy's chosen reparameterization noise eps, bounded to the
        # range the frozen decoder saw in training. Full derivations + measured
        # failure modes live in networks_kl_anchor.py; keep the math identical.
        if self.residual_mode == "sigma_ball":
            # Norm-bounded eps (soft-clip ||eps|| -> r, direction preserved) in
            # the prior's own sigma units — the per-dim tanh box let the policy
            # sit at joint radius k*sqrt(d), a corner N(0,I) never visits.
            # sqrt(sum + 1e-12) instead of jnp.linalg.norm: at raw == 0 (the
            # zero-init head) linalg.norm has a NaN gradient that would poison
            # the first update; with the epsilon the map is ~identity near 0
            # and residual is still exactly 0 at init.
            sigma_prior = jnp.exp(0.5 * z_logvar)
            r = self.residual_scale
            sq = jnp.sum(raw * raw, axis=-1, keepdims=True)
            norm = jnp.sqrt(sq + 1e-12)
            eps = raw * (r * jnp.tanh(norm / r) / norm)
            residual = sigma_prior * eps
        elif self.residual_mode == "sigma_tanh":
            sigma_prior = jnp.exp(0.5 * z_logvar)
            residual = self.residual_scale * sigma_prior * jnp.tanh(raw)
        elif self.residual_mode == "tanh":
            residual = self.residual_scale * jnp.tanh(raw)
        elif self.residual_mode == "none":
            residual = raw
        else:
            raise ValueError(
                f"unknown residual_mode {self.residual_mode!r}; "
                "expected one of 'sigma_ball', 'sigma_tanh', 'tanh', 'none'"
            )
        z = z_prior + residual

        decoder_input = jnp.concatenate([z, proprio], axis=-1)
        decoder_out = self.decoder(decoder_input)
        mu = decoder_out[..., : self.action_size]
        log_std = decoder_out[..., self.action_size :]
        # softplus + 1e-3 matches brax NormalTanhDistribution, which the imit
        # decoder's log_std head was trained against — required for the
        # warm-start splice to be bit-identical (see networks_kl_anchor.py).
        scale = jax.nn.softplus(log_std) + 1e-3
        return mu, scale, new_hidden

    def __call__(self, obs, hidden):
        mu, scale, new_hidden = self.raw(obs, hidden)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale), new_hidden


def make_dmpo_kl_anchor_rnn_networks(
    *,
    proprio_size: int,
    task_obs_size: int,
    action_size: int,
    latent_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    prior_layer_sizes: Sequence[int],
    decoder_layer_sizes: Sequence[int],
    rnn_cell: str = "gru",
    rnn_mlp_layers: Sequence[int] = (256,),
    rnn_hidden_sizes: Sequence[int] = (256,),
    rnn_store_dtype: Any = "float16",
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    warm_start_prior_params: Optional[Dict] = None,
    warm_start_decoder_params: Optional[Dict] = None,
    residual_mode: str = "sigma_ball",
    residual_scale: float = 5.0,
    critic_use_proprio: bool = False,
) -> DMPONetworks:
    """Build recurrent B-aggressive DMPO policy + FF critic for kl-anchor mode.

    Mirrors `make_dmpo_kl_anchor_networks` (same `_ValueVisionCriticNet`
    critic, same warm-start splice into prior/decoder) with the FF head's
    `policy_head_layer_sizes` replaced by the recurrent head config:
    `rnn_mlp_layers` (pre-RNN Dense+LayerNorm+silu mixing stage),
    `rnn_hidden_sizes` (stacked GRU cells), `rnn_cell` (v1: "gru" only),
    and `rnn_store_dtype` (replay dtype for the stored per-step hidden;
    accepts a dtype or its string name — hydra configs pass strings).

    The critic stays feed-forward on purpose: matches the recurrent-PPO
    precedent; Q's memory limitation is a documented v1 tradeoff, not an
    accident.

    Returns:
        `DMPONetworks(policy, critic, recurrent_meta)` — recurrent_meta is
        the trace-time switch rollout/learner/eval branch on.
    """
    del proprio_size, task_obs_size  # unused directly; consumed via obs dict.

    if rnn_cell != "gru":
        raise ValueError(
            f"policy_head_rnn cell {rnn_cell!r} is not supported in v1; "
            "only 'gru' (the tuple-of-arrays hidden contract has no room "
            "for LSTM (c, h) pairs)."
        )

    policy = _BAggressiveRecurrentPolicyNet(
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=tuple(prior_layer_sizes),
        decoder_layer_sizes=tuple(decoder_layer_sizes),
        rnn_mlp_layers=tuple(rnn_mlp_layers),
        rnn_hidden_sizes=tuple(rnn_hidden_sizes),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        rnn_cell=rnn_cell,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
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

    _orig_init = policy.init

    def policy_init(rng, obs, hidden):
        params = _orig_init(rng, obs, hidden)
        return _splice_warm_start(
            params, warm_start_prior_params, warm_start_decoder_params
        )

    policy.init = policy_init  # type: ignore[method-assign]

    return DMPONetworks(policy=policy, critic=critic, recurrent_meta=meta)
