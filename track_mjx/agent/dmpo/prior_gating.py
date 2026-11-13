"""Learned behavior/motion-prior gate for DMPO (``prior_gating:``), a port of SCAMPER-main's ``prior_gating`` mode.

SCAMPER (``scamper.agent.task_transfer.arena_forage.wrappers.PriorRGMDecoderHighLevelWrapper``) appends one scalar to
the policy's action: a gate value. A hard threshold with hysteresis turns it into a binary gate:

    gate = 1 if prev_gate == 0 and g > bias + margin  (switch on)
    gate = 0 if prev_gate == 1 and g < bias - margin  (switch off)
    otherwise gate = prev_gate

gate = 1 runs the policy as usual (motion-prior residual control); gate = 0 runs the BEHAVIOR prior instead,
``z = mu_BP(proprio) + exp(bp_logvar / 2) * eps``, decoded by the frozen decoder. The RL algorithm learns the gate from
reward like any other action dimension. ``prev_gate`` starts at 0, so with SCAMPER's bias 0.5 / margin 0.2 and a
zero-mean gate output the agent starts mostly on the behavior prior.

Here the gate lives env-side (``PriorGatingWrapper``), wrapped around either high-level wrapper:

* ``kind="motor"`` (around ``KLAnchorPriorDecoderWrapper``): the policy's action is [38-D motor, gate]. On gate = 0 the
  motor command is replaced by ``tanh(mu_decoder([z_BP, proprio]))``.
* ``kind="latent"`` (around ``LatentActionWrapper``): the policy's action is [16-D latent delta, gate]. On gate = 0 the
  delta is replaced by ``(z_BP - mu_MP) / scale``, so the wrapped LatentActionWrapper decodes exactly z_BP.

The stored action is always the policy's own [action, gate] sample (as in SCAMPER's PPO): on gate = 0 steps its
non-gate part had no effect, and the critic must learn that.

One deliberate deviation from SCAMPER: with ``gate_obs=True`` the CURRENT gate state (the ``prev_gate`` the next step's
hysteresis will read) is appended to ``obs['imitation_target']``, which both the policy trunk and the critic read. The
gate's effect depends on that state, and DMPO's critic Q(s, a) is only well defined if it can see it. SCAMPER's PPO
leaves it hidden.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _BAggressivePolicyNet,
    _PolicyHeadModule,
    _splice_warm_start,
)
from track_mjx.agent.dmpo.networks_vision_bottleneck import _ValueVisionCriticNet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

tfd = tfp.distributions

__all__ = [
    "GatedMotorPolicyNet",
    "PriorGatingWrapper",
    "hysteresis_gate",
    "make_bp_inference_fn",
    "make_dmpo_gated_motor_networks",
]

GATE_KINDS = ("motor", "latent")


def hysteresis_gate(prev_gate, g, bias: float, margin: float):
    """SCAMPER's bistable threshold on the (bound) gate action ``g``; returns 0./1. with prev_gate's shape."""
    upper, lower = bias + margin, bias - margin
    return jnp.where(
        prev_gate > 0.5,
        jnp.where(g < lower, 0.0, 1.0),
        jnp.where(g > upper, 1.0, 0.0),
    ).astype(jnp.float32)


def make_bp_inference_fn(
    behavior_prior_params, normalizer_params, latent_size: int, bp_layer_sizes
):
    """``(raw proprio) -> (mean, logvar)`` of the behavior prior, through the same SCAMPER inference code as the
    motion-prior ``prior_fn`` (``prior_utils.make_prior_inference_fn``), normalising with the checkpoint's stats."""
    from track_mjx.agent.dmpo.prior_utils import make_prior_inference_fn

    cfg = {
        "network_config": {
            "intention_size": int(latent_size),
            "prior_layer_sizes": list(bp_layer_sizes),
        }
    }
    return make_prior_inference_fn(behavior_prior_params, normalizer_params, cfg)


def _raw_proprio(state, policy_obs_key: str, lowlvl_obs_key: str):
    """The raw (un-normalised) proprio the frozen prior/decoder functions take, as the inner wrappers read it."""
    full = state.info.get("_full_obs", None) if isinstance(state.info, dict) else None
    if isinstance(full, dict):
        node = (
            full[policy_obs_key][lowlvl_obs_key]
            if policy_obs_key in full
            else full[lowlvl_obs_key]
        )
    else:
        node = state.obs["proprioception"]
    return jnp.nan_to_num(jax.flatten_util.ravel_pytree(node)[0])


def _base_wrapper_cls():
    from mujoco_playground._src import wrapper

    return wrapper.Wrapper


class PriorGatingWrapper:
    """Factory shim (like ``LatentActionWrapper``): returns a Wrapper subclass instance around ``env``.

    Args:
        env: the high-level wrapper (``KLAnchorPriorDecoderWrapper`` for kind="motor", ``LatentActionWrapper`` for
            kind="latent"). Its ``action_size`` is the non-gate action size.
        kind: "motor" or "latent".
        prior_fn / bp_fn / decoder_logits_fn: ``(raw proprio) -> (mean, logvar)`` motion / behavior prior, and the
            decoder logits fn (``prior_utils`` makers).
        latent_size, motor_size: 16 / 38 for the rat.
        latent_scale: the LatentActionWrapper's scale (kind="latent" only).
        gate_bias, gate_margin: SCAMPER defaults 0.5 / 0.2.
        bp_logvar: fixed log-variance of the behavior-prior latent sample (SCAMPER -2.25); None = its own prediction.
        gate_obs: append the current gate state to obs['imitation_target'].
    """

    def __new__(
        cls,
        env,
        *,
        kind: str,
        prior_fn: Callable,
        bp_fn: Callable,
        decoder_logits_fn: Callable,
        latent_size: int,
        motor_size: int,
        latent_scale: float = 1.0,
        gate_bias: float = 0.5,
        gate_margin: float = 0.2,
        bp_logvar: float | None = -2.25,
        gate_obs: bool = True,
        policy_obs_key: str = "state",
        lowlvl_obs_key: str = "proprioception",
    ):
        if kind not in GATE_KINDS:
            raise ValueError(
                f"prior_gating kind={kind!r}; expected one of {GATE_KINDS}"
            )
        base = _base_wrapper_cls()

        class _PriorGatingWrapper(base):
            def __init__(self):
                super().__init__(env)
                self._kind = kind
                self._inner_action_size = int(env.action_size)
                self._latent_size = int(latent_size)
                self._motor_size = int(motor_size)
                self._scale = float(latent_scale)
                self._bias, self._margin = float(gate_bias), float(gate_margin)
                self._bp_std = (
                    None
                    if bp_logvar is None
                    else float(jnp.exp(0.5 * float(bp_logvar)))
                )
                self._gate_obs = bool(gate_obs)
                expect = self._latent_size if kind == "latent" else self._motor_size
                if self._inner_action_size != expect:
                    raise ValueError(
                        f"prior_gating kind={kind}: inner action_size {self._inner_action_size} != {expect}"
                    )

            def _bp_action(self, state, key):
                """The inner-wrapper action that makes this step run the behavior prior."""
                proprio = _raw_proprio(state, policy_obs_key, lowlvl_obs_key)
                mu_b, lv_b = bp_fn(proprio)
                std = jnp.exp(0.5 * lv_b) if self._bp_std is None else self._bp_std
                z = mu_b + std * jax.random.normal(key, mu_b.shape, mu_b.dtype)
                if self._kind == "motor":
                    logits, _ = decoder_logits_fn(
                        jnp.concatenate([z, proprio], axis=-1)
                    )
                    return jnp.tanh(logits[..., : self._motor_size])
                mu_m, _ = prior_fn(proprio)
                return (z - mu_m) / self._scale

            def _with_gate(self, st, gate, g_action, inner=None):
                obs = st.obs
                if self._gate_obs:
                    obs = dict(obs)
                    obs["imitation_target"] = jnp.concatenate(
                        [
                            obs["imitation_target"],
                            jnp.reshape(gate, (1,)).astype(
                                obs["imitation_target"].dtype
                            ),
                        ],
                        axis=-1,
                    )
                m = dict(st.metrics) if st.metrics else {}
                m["gate/on"] = jnp.asarray(gate, jnp.float32)
                m["gate/action"] = jnp.asarray(g_action, jnp.float32)
                # Range diagnostics of the POLICY's own (bound) non-gate action, counted on gate-on steps only:
                # fraction of dims within 0.05 of the +-1 bound, and mean |a|. Summed per episode by the eval.
                if inner is None:
                    sat = absm = jnp.float32(0.0)
                else:
                    on = (gate > 0.5).astype(jnp.float32)
                    sat = on * jnp.mean((jnp.abs(inner) > 0.95).astype(jnp.float32))
                    absm = on * jnp.mean(jnp.abs(inner))
                m["gate/inner_sat_on"] = jnp.asarray(sat, jnp.float32)
                m["gate/inner_absmean_on"] = jnp.asarray(absm, jnp.float32)
                return st.replace(obs=obs, metrics=m)

            def reset(self, rng, **kwargs):
                st = self.env.reset(rng, **kwargs)
                st.info["prev_gate"] = jnp.float32(0.0)
                st.info["gate_rng"] = jax.random.fold_in(rng, 0x6A7E)
                return self._with_gate(st, jnp.float32(0.0), jnp.float32(0.0))

            def step(self, state, action):
                inner, g = (
                    action[..., : self._inner_action_size],
                    action[..., self._inner_action_size],
                )
                gate = hysteresis_gate(
                    state.info["prev_gate"], g, self._bias, self._margin
                )
                key, sub = jax.random.split(state.info["gate_rng"])
                a = jnp.where(gate > 0.5, inner, self._bp_action(state, sub))
                nxt = self.env.step(state, a)
                nxt.info["prev_gate"] = gate
                nxt.info["gate_rng"] = key
                return self._with_gate(nxt, gate, g, inner)

            @property
            def action_size(self) -> int:
                return self._inner_action_size + 1

            @property
            def observation_size(self):
                obs = dict(self.env.observation_size)
                if self._gate_obs:
                    obs["imitation_target"] = int(obs.get("imitation_target", 0)) + 1
                return obs

        return _PriorGatingWrapper()


class GatedMotorPolicyNet(_BAggressivePolicyNet):
    """The motor-action policy (frozen prior + residual head + frozen decoder -> 38-D Gaussian) plus one gate
    dimension: a zero-init ``gate_head`` (its own vision/task/proprio trunk) for the mean and a learnable
    state-independent std. Returns a 39-D ``MultivariateNormalDiag`` [motor..., gate]."""

    gate_init_std: float = 0.7

    def setup(self):
        super().setup()
        self.gate_head = _PolicyHeadModule(
            layer_sizes=tuple(self.policy_head_layer_sizes),
            latent_size=1,
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            name="gate_head",
        )
        import math

        init = math.log(math.expm1(float(self.gate_init_std) - 1e-3))
        self.gate_raw_std = self.param(
            "gate_raw_std", nn.initializers.constant(init), (1,)
        )

    def __call__(self, obs):
        motor = super().__call__(obs)
        g_mu = self.gate_head(
            obs["vision"], obs["imitation_target"], obs["proprioception"]
        )
        g_sd = jnp.broadcast_to(jax.nn.softplus(self.gate_raw_std) + 1e-3, g_mu.shape)
        return tfd.MultivariateNormalDiag(
            loc=jnp.concatenate([motor.mean(), g_mu], axis=-1),
            scale_diag=jnp.concatenate([motor.stddev(), g_sd], axis=-1),
        )


def make_dmpo_gated_motor_networks(
    *,
    motor_size: int,
    latent_size: int,
    vision_shape: tuple,
    cfg: Any,
    prior_layer_sizes: Sequence[int],
    decoder_layer_sizes: Sequence[int],
    policy_head_layer_sizes: Sequence[int] = (256, 256, 256),
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    warm_start_prior_params: dict | None = None,
    warm_start_decoder_params: dict | None = None,
    residual_mode: str = "sigma_tanh",
    residual_scale: float = 2.0,
    critic_use_proprio: bool = False,
    critic_privileged_dim: int = 0,
    gate_init_std: float = 0.7,
) -> DMPONetworks:
    """``make_dmpo_kl_anchor_networks`` with the policy swapped for ``GatedMotorPolicyNet`` (same critic)."""
    policy = GatedMotorPolicyNet(
        action_size=int(motor_size),
        latent_size=int(latent_size),
        vision_shape=tuple(vision_shape),
        prior_layer_sizes=tuple(prior_layer_sizes),
        decoder_layer_sizes=tuple(decoder_layer_sizes),
        policy_head_layer_sizes=tuple(policy_head_layer_sizes),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        gate_init_std=float(gate_init_std),
    )
    critic = _ValueVisionCriticNet(
        layer_sizes=tuple(value_hidden_layer_sizes),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        vision_shape=tuple(vision_shape),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        use_proprio=critic_use_proprio,
        privileged_dim=critic_privileged_dim,
    )
    _orig_init = policy.init

    def policy_init(rng, obs):
        return _splice_warm_start(
            _orig_init(rng, obs), warm_start_prior_params, warm_start_decoder_params
        )

    policy.init = policy_init  # type: ignore[method-assign]
    return DMPONetworks(policy=policy, critic=critic)
