"""Slow-clock memory core variant of the B-aggressive kl-anchor DMPO networks.

The FF kl-anchor policy head (`_PolicyHeadModule`) plus a GRU or LSTM "slow
core" that updates only every `tick_stride` env steps and holds its state in
between. The slow state is appended LAST to the FF head's MLP input, so the
head is the FF head with `hidden_0/kernel` widened by `slow_hidden_size`
trailing rows and one extra subtree, `slow_core`:

  * Graft: copy every leaf of an FF kl-anchor checkpoint and zero-fill the
    extra `hidden_0` rows, and the policy equals the FF policy for ANY
    hidden state -- slow_h only reaches the output through those rows.
    Param names therefore mirror `_PolicyHeadModule` exactly: auto-named
    `BinocularVisionEncoder_0`, `hidden_i` / `LayerNorm_i`, zero-init
    `residual`. The cell is named explicitly ("slow_core") so it cannot
    shift an auto-name.
  * Clock: the hidden is the tuple `(slow_h, phase)`, phase a [1] float
    leaf. The core ticks when phase < 0.5 and phase advances mod
    tick_stride. The rollout zeroes every hidden leaf on done, so an episode
    reset restarts the clock with a tick on the new episode's first step.
    tick_stride=1 ticks every step (a plain per-step cell feeding the FF MLP).
  * Cell types (`rnn_cell`): the hidden is a 2-leaf tuple for BOTH, so the
    rollout, replay, strided learner and eval need no change.
      - "gru":  `slow_h` is [.., H].
      - "lstm": `slow_h` is [.., 2H] = concat([c, h], -1), cell state FIRST,
        output second. Both halves are held on non-tick steps. The MLP reads
        only the H-wide output h (appended LAST), never c, so the head width
        and the FF graft (zero-filled `hidden_0` rows) are identical to the
        GRU's. `rnn_store_dtype` (float16 by default) is applied to the
        whole `slow_h` leaf. Unlike the GRU's h (in [-1, 1]), c is not
        squashed (only the emitted h = o * tanh(c) is), so at f16 it is
        range-limited (|c| <= 65504) and rounded to ~3 decimal digits in
        replay. The live rollout carry stays f32; only the replay copy is
        rounded.

Prior and Decoder are the modules from networks_kl_anchor (warm-start splice
unchanged) and the head stays under `policy_head`, so
`optim_kl_anchor.label_param_tree` gives the slow core the policy_head LR.
"""

from __future__ import annotations

# Runtime import despite `from __future__ import annotations`: this module's
# Flax Module dataclass annotations are resolved at runtime by
# typing.get_type_hints() (e.g. the optional beartype runtime-checker
# extra), so a TYPE_CHECKING-only import raises NameError even though flax
# init/apply/clone never need it resolved.
from collections.abc import Sequence  # noqa: TC003
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

# Runtime import for the same reason as the Sequence import above.
from track_mjx.agent.dmpo.config import DMPOConfig  # noqa: TC001
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _DecoderModule,
    _PriorModule,
    _splice_warm_start,
)
from track_mjx.agent.dmpo.networks_kl_anchor import (
    latent_residual as _latent_residual,
)
from track_mjx.agent.dmpo.networks_vision_bottleneck import (
    _ValueVisionCriticNet,
)
from track_mjx.agent.dmpo.vision_encoders import BinocularVisionEncoder

tfd = tfp.distributions

# Packed width of the slow_h leaf per cell, in units of the core width H:
# a GRU carries h; an LSTM carries concat([c, h]).
_STATE_WIDTH_MULT = {"gru": 1, "lstm": 2}


class SlowCoreMeta(NamedTuple):
    """Static description of the slow-core hidden state `(slow_h, phase)`.

    Drop-in for `RecurrentPolicyMeta` (rollout / eval / training loop only
    use `init_hidden` and `store_dtype`; `cell_type` / `hidden_sizes` are
    read for logging). `hidden_sizes[0]` is the core width H (the width of
    the output the FF head reads); the stored `slow_h` leaf is
    `state_size` wide: H for a GRU, 2H (`concat([c, h])`) for an LSTM.
    `tick_stride` is the core's update period in env steps; `phase_leaf` is
    the index of the phase leaf in the hidden tuple, which the strided
    learner reads to find tick positions.
    """

    cell_type: str
    hidden_sizes: tuple
    store_dtype: Any
    tick_stride: int
    phase_leaf: int = 1

    @property
    def state_size(self) -> int:
        """Width of the `slow_h` leaf: H (gru) or 2H (lstm, `[c, h]`)."""
        return self.hidden_sizes[0] * _STATE_WIDTH_MULT[self.cell_type]

    def init_hidden(self, batch_size: int | None = None) -> tuple:
        """Zero `(slow_h, phase)`: `([B, W], [B, 1])`, or `([W], [1])`
        unbatched, with W = `state_size` (H for gru, 2H for lstm)."""
        lead = () if batch_size is None else (batch_size,)
        return (
            jnp.zeros((*lead, self.state_size)),
            jnp.zeros((*lead, 1)),
        )


class _SlowCorePolicyHeadModule(nn.Module):
    """CNN + task_obs + proprio (+ held slow GRU/LSTM state) -> residual.

    Creation order fixes the param names: encoder (auto
    `BinocularVisionEncoder_0`), cell `slow_core`, `hidden_i` / `LayerNorm_i`,
    zero-init `residual`. Shape-agnostic over leading dims (unbatched is the
    production path: the rollout and learner vmap over envs).

    `cell_type="gru"`: `slow_h` is [.., H]. `cell_type="lstm"`: `slow_h` is
    [.., 2H] = `concat([c, h], -1)`; the cell runs on the carry `(c, h)` and
    BOTH halves are held on non-tick steps. In either case the MLP reads
    `concat([x, h])` with the H-wide output h LAST (never c). `c` is not
    squashed; the replay store dtype applies to the whole leaf.
    """

    layer_sizes: Sequence[int]
    slow_hidden_size: int
    tick_stride: int
    latent_size: int
    vision_shape: tuple
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    cell_type: str = "gru"

    @nn.compact
    def __call__(self, vision, task_obs, proprio, hidden):
        slow_h, phase = hidden
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(vision)
        x = jnp.concatenate([vis, task_obs, proprio], axis=-1)

        # The core reads the tick frame only. It is evaluated every step and
        # selected by the tick mask (a cond would become a select under the
        # per-env vmap anyway). kernel_init matches the repo's recurrent heads
        # (recurrent_ppo.get_rnn_cell), which cannot take an explicit name.
        tick = phase[..., :1] < 0.5  # [..., 1]; broadcasts over the leaf width
        H = self.slow_hidden_size
        if self.cell_type == "gru":
            g, _ = nn.GRUCell(
                features=H,
                kernel_init=jax.nn.initializers.lecun_uniform(),
                name="slow_core",
            )(slow_h, x)
            slow_h = jnp.where(tick, g, slow_h)
            out = slow_h
        elif self.cell_type == "lstm":
            # slow_h = concat([c, h]); one where() over the packed leaf holds
            # BOTH c and h on non-tick steps.
            carry = (slow_h[..., :H], slow_h[..., H:])
            (new_c, new_h), _ = nn.LSTMCell(
                features=H,
                kernel_init=jax.nn.initializers.lecun_uniform(),
                name="slow_core",
            )(carry, x)
            slow_h = jnp.where(tick, jnp.concatenate([new_c, new_h], axis=-1), slow_h)
            out = slow_h[..., H:]  # the output h only -- never c
        else:
            raise ValueError(
                f"unknown slow-core cell_type {self.cell_type!r}; "
                "expected 'gru' or 'lstm'"
            )

        h = jnp.concatenate([x, out], axis=-1)  # output LAST -> graftable
        for i, size in enumerate(self.layer_sizes):
            h = nn.Dense(size, name=f"hidden_{i}")(h)
            h = nn.LayerNorm(name=f"LayerNorm_{i}")(h)
            h = nn.silu(h)
        residual = nn.Dense(
            self.latent_size,
            name="residual",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        new_phase = jnp.mod(phase + 1.0, float(self.tick_stride))
        return residual, (slow_h, new_phase)


class _BAggressiveSlowCorePolicyNet(nn.Module):
    """Trainable Prior + slow-core PolicyHead + Decoder.

    Same apply paths as `_BAggressiveRecurrentPolicyNet`:
      - `__call__(obs, hidden) -> (tfd.MultivariateNormalDiag, new_hidden)`;
      - `raw(obs, hidden) -> (mu, scale, new_hidden)` (plain arrays, for the
        learner's `lax.scan`).
    """

    action_size: int
    latent_size: int
    vision_shape: tuple
    prior_layer_sizes: Sequence[int]
    decoder_layer_sizes: Sequence[int]
    policy_head_layer_sizes: Sequence[int]
    slow_hidden_size: int
    tick_stride: int
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    residual_mode: str = "sigma_ball"
    residual_scale: float = 5.0
    cell_type: str = "gru"

    def setup(self):
        self.prior = _PriorModule(
            layer_sizes=tuple(self.prior_layer_sizes),
            latents=self.latent_size,
            name="prior",
        )
        self.decoder = _DecoderModule(
            layer_sizes=(*tuple(self.decoder_layer_sizes), 2 * self.action_size),
            name="decoder",
        )
        self.policy_head = _SlowCorePolicyHeadModule(
            layer_sizes=tuple(self.policy_head_layer_sizes),
            slow_hidden_size=self.slow_hidden_size,
            tick_stride=self.tick_stride,
            latent_size=self.latent_size,
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            cell_type=self.cell_type,
            name="policy_head",
        )

    def raw(self, obs, hidden):
        proprio = obs["proprioception"]
        z_prior, z_logvar = self.prior(proprio)
        head_out, new_hidden = self.policy_head(
            obs["vision"], obs["imitation_target"], proprio, hidden
        )
        z = z_prior + _latent_residual(
            head_out, z_logvar, self.residual_mode, self.residual_scale
        )
        decoder_out = self.decoder(jnp.concatenate([z, proprio], axis=-1))
        mu = decoder_out[..., : self.action_size]
        # softplus + 1e-3 matches the imitation decoder's training transform
        # (brax NormalTanhDistribution); see networks_kl_anchor.py.
        scale = jax.nn.softplus(decoder_out[..., self.action_size :]) + 1e-3
        return mu, scale, new_hidden

    def __call__(self, obs, hidden):
        mu, scale, new_hidden = self.raw(obs, hidden)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale), new_hidden


def make_dmpo_kl_anchor_slow_networks(
    *,
    proprio_size: int,
    task_obs_size: int,
    action_size: int,
    latent_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    prior_layer_sizes: Sequence[int],
    decoder_layer_sizes: Sequence[int],
    policy_head_layer_sizes: Sequence[int] = (256, 256, 256),
    slow_hidden_size: int = 512,
    tick_stride: int = 10,
    rnn_cell: str = "gru",
    rnn_store_dtype: Any = "float16",
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    warm_start_prior_params: dict | None = None,
    warm_start_decoder_params: dict | None = None,
    residual_mode: str = "sigma_ball",
    residual_scale: float = 5.0,
    critic_use_proprio: bool = False,
    critic_privileged_dim: int = 0,
) -> DMPONetworks:
    """Build the slow-core B-aggressive DMPO policy + FF critic (kl-anchor).

    Shares every kwarg of `make_dmpo_kl_anchor_rnn_networks` except the head
    block: `policy_head_layer_sizes` is the FF head's MLP (same meaning as in
    `make_dmpo_kl_anchor_networks`, so an FF checkpoint grafts in), plus
    `slow_hidden_size` (core width H) and `tick_stride` (core update period
    k in env steps). `rnn_cell` is "gru" or "lstm". The hidden is a 2-leaf
    tuple `(slow_h, phase)` either way: `slow_h` is [H] for a GRU and
    [2H] = `concat([c, h])` for an LSTM (the FF head reads only h; see the
    module docstring). `rnn_store_dtype` is the replay dtype of the stored
    hidden, both leaves -- it must hold every phase value 0..k-1 exactly
    (float16: k <= 2049); for an LSTM it also stores the cell state c,
    which float16 limits to |c| <= 65504 at ~3 digits.

    Like the recurrent factory, `policy.init(rng, obs, hidden)` takes a
    hidden (use `recurrent_meta.init_hidden()`), and the warm-start
    prior/decoder are spliced after init.

    Returns:
        `DMPONetworks(policy, critic, recurrent_meta=SlowCoreMeta(...))`.
    """
    del proprio_size, task_obs_size  # unused directly; consumed via obs dict.

    if rnn_cell not in _STATE_WIDTH_MULT:
        raise ValueError(
            f"slow core cell {rnn_cell!r} is not supported; expected 'gru' or 'lstm'."
        )
    tick_stride = int(tick_stride)
    if tick_stride < 1:
        raise ValueError(f"tick_stride must be >= 1, got {tick_stride}.")
    store_dtype = jnp.dtype(rnn_store_dtype)
    phases = np.arange(tick_stride, dtype=np.float64)
    if not np.array_equal(phases.astype(store_dtype).astype(np.float64), phases):
        raise ValueError(
            f"tick_stride={tick_stride}: rnn_store_dtype {store_dtype} cannot "
            f"hold every phase 0..{tick_stride - 1} exactly, so replay would "
            "store corrupted tick positions. Use a wider store dtype or a "
            "smaller tick_stride."
        )

    policy = _BAggressiveSlowCorePolicyNet(
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=tuple(prior_layer_sizes),
        decoder_layer_sizes=tuple(decoder_layer_sizes),
        policy_head_layer_sizes=tuple(policy_head_layer_sizes),
        slow_hidden_size=int(slow_hidden_size),
        tick_stride=tick_stride,
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        cell_type=rnn_cell,
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
        privileged_dim=critic_privileged_dim,
    )

    meta = SlowCoreMeta(
        cell_type=rnn_cell,
        hidden_sizes=(int(slow_hidden_size),),
        store_dtype=store_dtype,
        tick_stride=tick_stride,
    )

    _orig_init = policy.init

    def policy_init(rng, obs, hidden):
        params = _orig_init(rng, obs, hidden)
        return _splice_warm_start(
            params, warm_start_prior_params, warm_start_decoder_params
        )

    policy.init = policy_init  # type: ignore[method-assign]

    return DMPONetworks(policy=policy, critic=critic, recurrent_meta=meta)
