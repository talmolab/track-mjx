"""Adversarial Differential Discriminator utilities for motion tracking.

This module contains the shared pieces needed to train ADD-style rewards on top
of the existing PPO pipelines:

- an environment wrapper that exports per-step tracking differentials in
  ``state.info["add_differential"]``;
- a small discriminator network factory;
- discriminator reward and loss functions.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import brax.math
import jax
import jax.numpy as jnp
from brax.training import networks
from brax.training.types import Params
from mujoco import mjx
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

try:
    from vnl_playground.tasks.rodent import consts as rodent_consts
except ImportError:  # pragma: no cover - vnl_playground is a runtime dependency.
    rodent_consts = None


DEFAULT_FEATURE_GROUPS = {
    "root_pos": True,
    "root_rot": True,
    "root_vel": True,
    "root_ang_vel": True,
    "joints": True,
    "joints_vel": True,
    "end_eff": True,
    "bodies_pos": False,
}

DEFAULT_FEATURE_SCALES = {
    "root_pos": 0.035,
    "root_rot": 0.35,
    "root_vel": 1.0,
    "root_ang_vel": 1.0,
    "joints": 1.41,
    "joints_vel": 1.0,
    "end_eff": 0.03,
    "bodies_pos": 0.25,
}


def config_get(config: Mapping[str, Any] | Any | None, key: str, default: Any) -> Any:
    """Read a key from dict-like or attribute configs."""
    if config is None:
        return default
    if isinstance(config, Mapping) and key in config:
        return config[key]
    return getattr(config, key, default)


def is_enabled(add_config: Mapping[str, Any] | Any | None) -> bool:
    """Returns whether ADD is enabled in config."""
    return bool(config_get(add_config, "enabled", False))


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(v) for v in value)


def _scaled(name: str, value: jnp.ndarray, scales: Mapping[str, float]) -> jnp.ndarray:
    return jnp.ravel(value / float(scales[name]))


class ADDDifferentialWrapper(wrapper.Wrapper):
    """Adds ADD tracking differentials to ``state.info``.

    The wrapper assumes the wrapped environment exposes the imitation helper
    methods used by VNL imitation tasks, for example ``_get_current_target``,
    ``root_body``, ``_get_joint_angles``, and ``_get_bodies_pos``.
    """

    def __init__(self, env: mjx_env.MjxEnv, add_config: Mapping[str, Any] | Any):
        super().__init__(env)
        feature_groups = dict(DEFAULT_FEATURE_GROUPS)
        feature_groups.update(config_get(add_config, "feature_groups", {}) or {})
        feature_scales = dict(DEFAULT_FEATURE_SCALES)
        feature_scales.update(config_get(add_config, "feature_scales", {}) or {})

        self._feature_groups = {
            name: bool(enabled) for name, enabled in feature_groups.items()
        }
        self._feature_scales = {
            name: float(scale) for name, scale in feature_scales.items()
        }
        for name, scale in self._feature_scales.items():
            if scale <= 0.0:
                raise ValueError(f"ADD feature scale '{name}' must be positive.")

        self._end_effector_bodies = _as_tuple(
            config_get(add_config, "end_effector_bodies", None)
        )
        self._body_pos_bodies = _as_tuple(
            config_get(add_config, "body_pos_bodies", None)
        )

        if not self._end_effector_bodies and rodent_consts is not None:
            self._end_effector_bodies = tuple(rodent_consts.END_EFFECTORS)
        if not self._body_pos_bodies and rodent_consts is not None:
            self._body_pos_bodies = tuple(rodent_consts.BODIES)

        if not any(self._feature_groups.values()):
            raise ValueError("ADD requires at least one enabled feature group.")

    def reset(self, rng: jax.Array, **kwargs: Any) -> mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        return self._augment_state(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        return self._augment_state(state)

    def _augment_state(self, state: mjx_env.State) -> mjx_env.State:
        differential, feature_norms = self._get_differential(state.data, state.info)
        differential = jnp.nan_to_num(differential)
        info = dict(state.info)
        info["add_differential"] = differential

        metrics = dict(state.metrics)
        metrics["add/differential_norm"] = jnp.linalg.norm(differential)
        for name, norm in feature_norms.items():
            metrics[f"add/{name}_norm"] = jnp.nan_to_num(norm)

        return state.replace(info=info, metrics=metrics)

    def _get_differential(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        target = self.env._get_current_target(data, info)
        root = self.env.root_body(data)
        scales = self._feature_scales
        features = []
        feature_norms: dict[str, jnp.ndarray] = {}

        def add_feature(name: str, diff: jnp.ndarray) -> None:
            scaled = _scaled(name, diff, scales)
            features.append(scaled)
            feature_norms[name] = jnp.linalg.norm(scaled)

        if self._feature_groups.get("root_pos", False):
            add_feature("root_pos", target.root_position - root.xpos)

        if self._feature_groups.get("root_rot", False):
            rel_quat = brax.math.relative_quat(target.root_quaternion, root.xquat)
            add_feature("root_rot", brax.math.quat_to_euler(rel_quat))

        if self._feature_groups.get("root_vel", False):
            add_feature(
                "root_vel",
                brax.math.rotate(target.velocity - data.qvel[:3], root.xquat),
            )

        if self._feature_groups.get("root_ang_vel", False):
            add_feature(
                "root_ang_vel",
                brax.math.rotate(target.angular_velocity - data.qvel[3:6], root.xquat),
            )

        if self._feature_groups.get("joints", False):
            add_feature("joints", target.joints - self.env._get_joint_angles(data))

        if self._feature_groups.get("joints_vel", False):
            add_feature(
                "joints_vel",
                target.joints_velocity - self.env._get_joint_ang_vels(data),
            )

        bodies_pos = None
        if self._feature_groups.get("end_eff", False):
            if not self._end_effector_bodies:
                raise ValueError(
                    "ADD end_eff feature is enabled but no bodies are set."
                )
            bodies_pos = self.env._get_bodies_pos(data, flatten=False)
            add_feature(
                "end_eff",
                jnp.concatenate(
                    [
                        target.body_xpos(body_name) - bodies_pos[body_name]
                        for body_name in self._end_effector_bodies
                    ],
                    axis=-1,
                ),
            )

        if self._feature_groups.get("bodies_pos", False):
            if not self._body_pos_bodies:
                raise ValueError(
                    "ADD bodies_pos feature is enabled but no bodies are set."
                )
            if bodies_pos is None:
                bodies_pos = self.env._get_bodies_pos(data, flatten=False)
            add_feature(
                "bodies_pos",
                jnp.concatenate(
                    [
                        target.body_xpos(body_name) - bodies_pos[body_name]
                        for body_name in self._body_pos_bodies
                    ],
                    axis=-1,
                ),
            )

        return jnp.concatenate(features, axis=-1), feature_norms


def make_discriminator_network(
    differential_size: int,
    hidden_layer_sizes: Sequence[int] = (1024, 512),
) -> networks.FeedForwardNetwork:
    """Creates a discriminator network over ADD differentials."""
    return networks.make_value_network(
        obs_size=differential_size,
        hidden_layer_sizes=tuple(hidden_layer_sizes),
    )


def discriminator_reward(
    discriminator_params: Params,
    differentials: jnp.ndarray,
    discriminator_network: networks.FeedForwardNetwork,
    reward_scale: float = 1.0,
) -> jnp.ndarray:
    """Computes the ADD policy reward from discriminator logits."""
    logits = discriminator_network.apply(None, discriminator_params, differentials)
    return jax.lax.stop_gradient(float(reward_scale) * jax.nn.softplus(logits))


def compute_discriminator_loss(
    discriminator_params: Params,
    differentials: jnp.ndarray,
    rng: jnp.ndarray,
    discriminator_network: networks.FeedForwardNetwork,
    grad_penalty_weight: float = 1.0,
    logit_reg_weight: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Computes discriminator BCE loss with zero positives and negative GP."""
    del rng
    differentials = jax.lax.stop_gradient(differentials)
    differentials = differentials.reshape((-1, differentials.shape[-1]))
    positives = jnp.zeros_like(differentials)

    pos_logits = discriminator_network.apply(None, discriminator_params, positives)
    neg_logits = discriminator_network.apply(None, discriminator_params, differentials)

    positive_loss = jnp.mean(jax.nn.softplus(-pos_logits))
    negative_loss = jnp.mean(jax.nn.softplus(neg_logits))
    bce_loss = positive_loss + negative_loss

    def disc_score_sum(inputs: jnp.ndarray) -> jnp.ndarray:
        logits = discriminator_network.apply(None, discriminator_params, inputs)
        return jnp.sum(jax.nn.sigmoid(logits))

    gp_loss = jnp.array(0.0, dtype=differentials.dtype)
    if grad_penalty_weight > 0.0:
        gradients = jax.grad(disc_score_sum)(differentials)
        gp_loss = float(grad_penalty_weight) * jnp.mean(
            jnp.sum(jnp.square(gradients), axis=-1)
        )

    logit_reg = jnp.array(0.0, dtype=differentials.dtype)
    if logit_reg_weight > 0.0:
        logit_reg = float(logit_reg_weight) * jnp.mean(
            jnp.square(pos_logits) + jnp.square(neg_logits)
        )

    total_loss = bce_loss + gp_loss + logit_reg
    pos_score = jnp.mean(jax.nn.sigmoid(pos_logits))
    neg_score = jnp.mean(jax.nn.sigmoid(neg_logits))

    return total_loss, {
        "disc_loss": total_loss,
        "disc_bce_loss": bce_loss,
        "gp_loss": gp_loss,
        "disc_gp_loss": gp_loss,
        "disc_logit_reg": logit_reg,
        "disc_score_pos": pos_score,
        "disc_score_neg": neg_score,
        "differential_norm": jnp.mean(jnp.linalg.norm(differentials, axis=-1)),
    }
