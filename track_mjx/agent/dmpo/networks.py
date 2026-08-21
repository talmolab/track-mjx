"""DMPO network heads in Flax linen.

Port of acme/jax/networks/distributional.py:
- MultivariateNormalDiagHead -> GaussianPolicyHead
- DiscreteValuedTfpHead     -> CategoricalCriticHead

Plus a `make_dmpo_networks` factory that wires policy + critic torsos to
the heads. Each torso block is Dense -> LayerNorm -> SiLU (matching
networks_vision.py and Acme's LayerNormMLP pattern). SiLU is used in
place of Acme's tanh+ELU for compatibility with the wider track-mjx
convention. The loss-relevant *heads* are byte-for-byte Acme.
"""
from typing import Any, Callable, NamedTuple, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

_MIN_SCALE = 1e-6


class GaussianPolicyHead(nn.Module):
    """Linen port of acme.jax.networks.MultivariateNormalDiagHead.

    Outputs an unbounded MultivariateNormalDiag. Action squashing (tanh) is
    NOT applied here - the MPO loss requires unbounded Gaussians for its KL
    decomposition. Use action_utils.bind / unbind at the env boundary.

    Mirrors Acme's exact scale formulation:
        scale = softplus(linear(x))
        scale *= init_scale / softplus(0.)
        scale += min_scale
    so that when the linear pre-activation is ~0 (zero-init weights and bias),
    scale_diag is approximately init_scale.
    """
    action_size: int
    init_scale: float = 0.7
    min_scale: float = _MIN_SCALE
    w_init: Callable = nn.initializers.variance_scaling(
        scale=1e-4, mode="fan_in", distribution="truncated_normal"
    )
    b_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        loc = nn.Dense(
            self.action_size,
            kernel_init=self.w_init,
            bias_init=self.b_init,
            name="loc",
        )(x)
        scale = nn.Dense(
            self.action_size,
            kernel_init=self.w_init,
            bias_init=self.b_init,
            name="scale",
        )(x)
        scale = jax.nn.softplus(scale)
        scale = scale * (self.init_scale / jax.nn.softplus(0.0))
        scale = scale + self.min_scale
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)


class DiscreteValuedTfpDistribution(tfd.Categorical):
    """Generalization of tfd.Categorical that knows its real-valued support.

    Port of acme.jax.networks.DiscreteValuedTfpDistribution. The support
    `values` can be any real-valued range (vs. [0, n-1] for plain Categorical),
    which lets us take a meaningful mean/variance over it. Used as the C51
    critic distribution in DMPO.
    """

    def __init__(
        self,
        values: jnp.ndarray,
        logits: Optional[jnp.ndarray] = None,
        probs: Optional[jnp.ndarray] = None,
        name: str = "DiscreteValuedDistribution",
    ):
        parameters = dict(locals())
        self._values = np.asarray(values)

        if logits is not None:
            logits = jnp.asarray(logits)
        if probs is not None:
            probs = jnp.asarray(probs)

        super().__init__(logits=logits, probs=probs, name=name)
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        """Declare parameter properties to silence the tfp inheritance warning.

        tfp warns whenever a Distribution subclass overrides ``__init__`` but
        inherits ``_parameter_properties`` from its parent — the parent's
        properties may not match the subclass's signature. We document
        ``logits`` (the only tensor-valued, batchable parameter) and ``values``
        (the static real-valued support, which is not a tensor and so gets
        ``is_tensor=False``). ``probs`` is omitted intentionally: callers
        always pass logits in our pipeline, and listing it here as a tensor
        parameter would trip tfp's "exactly one of logits/probs" guards.
        """
        return dict(
            logits=tfp.util.ParameterProperties(event_ndims=1),
            values=tfp.util.ParameterProperties(
                event_ndims=1,
                shape_fn=lambda sample_shape: sample_shape[-1:],
                is_tensor=False,
                specifies_shape=True,
            ),
        )

    @property
    def values(self) -> jnp.ndarray:
        return self._values

    def _sample_n(self, key, n):
        indices = super()._sample_n(key=key, n=n)
        return jnp.take_along_axis(self._values, indices, axis=-1)

    def mean(self) -> jnp.ndarray:
        """Mean using the real-valued support, not the integer indices."""
        return jnp.sum(self.probs_parameter() * self._values, axis=-1)

    def variance(self) -> jnp.ndarray:
        dist_squared = jnp.square(jnp.expand_dims(self.mean(), -1) - self._values)
        return jnp.sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self):
        return jnp.zeros((), dtype=jnp.int32)

    def _event_shape_tensor(self):
        return []


class CategoricalCriticHead(nn.Module):
    """Linen port of acme.jax.networks.DiscreteValuedTfpHead.

    Categorical critic over `num_atoms` fixed atoms uniformly spaced in
    [vmin, vmax]. This is the C51-style distributional critic head used by
    DMPO. The returned distribution exposes its support via `dist.values` and
    overrides `mean()` / `variance()` to incorporate it.

    Acme's haiku version stores the atoms as a private numpy attribute and
    exposes them via the returned distribution. The linen port additionally
    exposes them as a `@property` on the module itself, since the support is a
    deterministic function of the dataclass fields and is needed by the
    Bellman projection (Task 10) before `apply` is called.
    """

    num_atoms: int
    vmin: float
    vmax: float
    w_init: Optional[Callable] = None
    b_init: Optional[Callable] = None

    @property
    def values(self) -> np.ndarray:
        """Atom support, length `num_atoms`, spanning [vmin, vmax].

        Returned as a static numpy array (mirroring Acme); this keeps the
        atom support out of jax trace contexts so that downstream code
        like `DiscreteValuedTfpDistribution.__init__` (which calls
        `np.asarray(values)`) doesn't fail under `jax.jit`.
        """
        return np.linspace(self.vmin, self.vmax, num=self.num_atoms)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        # Match Acme: pass init kwargs only if provided, so nn.Dense defaults
        # apply otherwise (matching Haiku Linear's default behavior with
        # w_init=None / b_init=None).
        dense_kwargs = {}
        if self.w_init is not None:
            dense_kwargs["kernel_init"] = self.w_init
        if self.b_init is not None:
            dense_kwargs["bias_init"] = self.b_init
        logits = nn.Dense(self.num_atoms, name="logits", **dense_kwargs)(inputs)
        return DiscreteValuedTfpDistribution(values=self.values, logits=logits)


# ---------------------------------------------------------------------------
# Task 4: DMPO network factory.
#
# Acme's `make_control_networks` (acme/agents/jax/mpo/networks.py) wraps the
# policy and critic in a `MPONetworks` dataclass with an unrollable `torso`
# (for recurrent backbones) and separate `policy_head` / `critic_head`
# `hk.Transformed`s sharing a torso embedding. We deliberately flatten this:
# DMPO's first cut is feed-forward only, so we wire each head directly behind
# its own torso. The recurrent torso plumbing can be reintroduced later as a
# separate module if needed; the current shape mirrors vnl-ray's
# `train_dmpo_ray.py` (separate policy and critic torsos).
# ---------------------------------------------------------------------------


class DMPONetworks(NamedTuple):
    """Bundle of (policy, critic) flax modules used by DMPO.

    `recurrent_meta` is a `RecurrentPolicyMeta` (networks_kl_anchor_rnn) when
    the policy is recurrent, else None — the trace-time switch that
    rollout/learner/eval branch on; the default keeps every existing 2-arg
    construction (and all FF behavior) unchanged.
    """

    policy: nn.Module
    critic: nn.Module
    recurrent_meta: Any = None


class _PolicyNet(nn.Module):
    """MLP torso (Dense -> LayerNorm -> SiLU per block) -> GaussianPolicyHead."""

    layer_sizes: Sequence[int]
    action_size: int
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tfd.Distribution:
        h = obs
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return GaussianPolicyHead(action_size=self.action_size)(h)


class _CriticNet(nn.Module):
    """MLP torso over concat([obs, action]) -> CategoricalCriticHead.

    Matches Acme's `critic_fn` in `make_control_networks`: action is appended
    on the last axis and the joint vector is fed through the torso. We omit
    Acme's `ClipToSpec` step because actions in track-mjx are already
    normalized to [-1, 1] at the env boundary (see `action_utils.bind`); if
    that invariant ever changes, add the clip here.
    """

    layer_sizes: Sequence[int]
    num_atoms: int
    vmin: float
    vmax: float
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> tfd.Distribution:
        h = jnp.concatenate([obs, action], axis=-1)
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


def make_dmpo_networks(
    obs_size: int, action_size: int, cfg
) -> DMPONetworks:
    """Build (policy, critic) flax modules for DMPO.

    Args:
        obs_size: Observation dimensionality. Currently unused in the body
            (only `action_size` and `cfg` shape the modules), but accepted in
            the signature for symmetry with how env specs are passed around
            elsewhere in track-mjx.
        action_size: Action dimensionality.
        cfg: A `DMPOConfig` (or anything exposing `policy_layer_sizes`,
            `critic_layer_sizes`, `num_atoms`, `vmin`, `vmax`).

    Returns:
        `DMPONetworks(policy, critic)` ready to be `init`'d with dummy obs/act.
    """
    del obs_size  # unused; see docstring.
    return DMPONetworks(
        policy=_PolicyNet(
            layer_sizes=tuple(cfg.policy_layer_sizes),
            action_size=action_size,
        ),
        critic=_CriticNet(
            layer_sizes=tuple(cfg.critic_layer_sizes),
            num_atoms=cfg.num_atoms,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
        ),
    )
