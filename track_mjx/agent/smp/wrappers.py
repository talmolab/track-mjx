"""VNL environment wrapper that adds SMP prior rewards."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground import wrapper

from track_mjx.agent.smp.features import SMPFeatureSpec, compute_smp_obs
from track_mjx.agent.smp.reward import (
    DiffNormalizer,
    SMPRewardConfig,
    blend_task_and_smp_rewards,
    compute_smp_reward,
)
from track_mjx.agent.smp.tinymdm import SMPNormalizer, TinyDiTDenoiser, TinyMDMConfig


class SMPRewardWrapper(wrapper.Wrapper):
    """Adds reward-only SMP to VNL rodent task environments.

    The wrapper keeps a fixed-length SMP feature history in ``state.info`` and
    replaces the task reward with a task/SMP weighted blend.  It is intended to
    sit inside the observation-flattening wrappers, directly around a VNL rodent
    environment.
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        prior_params: Any,
        prior_normalizer: SMPNormalizer,
        diff_normalizer: DiffNormalizer,
        model_config: TinyMDMConfig,
        feature_spec: SMPFeatureSpec,
        reward_config: SMPRewardConfig = SMPRewardConfig(),
        metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__(env)
        self._prior_params = prior_params
        self._prior_normalizer = prior_normalizer
        self._diff_normalizer = diff_normalizer
        self._model_config = model_config
        self._feature_spec = feature_spec
        self._reward_config = reward_config
        self._prior_model = TinyDiTDenoiser(model_config)
        self._metadata = dict(metadata or {})
        self._smp_stride = self._compute_smp_stride()
        self._validate_metadata()

    def _compute_smp_stride(self) -> int:
        ctrl_dt = float(getattr(self.env, "_config").ctrl_dt)
        target_dt = 1.0 / float(self._feature_spec.mocap_hz)
        return max(int(round(target_dt / ctrl_dt)), 1)

    def _validate_metadata(self) -> None:
        if not self._metadata:
            return
        expected = self._metadata.get("joint_names")
        if expected is not None and hasattr(self.env, "get_joint_names"):
            actual = list(self.env.get_joint_names())
            if not self._joint_names_match(list(expected), actual):
                raise ValueError(
                    "SMP prior joint names do not match task environment.\n"
                    f"  prior (first 3): {list(expected)[:3]}\n"
                    f"  env   (first 3): {actual[:3]}"
                )
        scale = self._metadata.get("scale_factor")
        env_scale = getattr(getattr(self.env, "_config", None), "rescale_factor", None)
        if (
            scale is not None
            and env_scale is not None
            and abs(float(scale) - env_scale) > 1e-6
        ):
            raise ValueError(
                f"SMP prior scale_factor={scale} does not match env rescale_factor={env_scale}."
            )
        feature_meta = self._metadata.get("feature_spec")
        if feature_meta is not None:
            prior_spec = SMPFeatureSpec.from_dict(feature_meta)
            if prior_spec != self._feature_spec:
                raise ValueError(
                    "Loaded SMP prior feature spec does not match wrapper spec."
                )

    @staticmethod
    def _joint_names_match(prior_names: list[str], env_names: list[str]) -> bool:
        if prior_names == env_names:
            return True
        if len(prior_names) != len(env_names):
            return False
        # vnl-playground appends a '-<walker>' suffix to env joint names
        # (e.g. 'knee_L' in clips becomes 'knee_L-rodent' in the env).
        stripped = [n.rsplit("-", 1)[0] for n in env_names]
        return stripped == prior_names

    def reset(self, rng: jax.Array, **kwargs: Any) -> wrapper.mjx_env.State:
        rng, reward_rng = jax.random.split(rng)
        state = self.env.reset(rng, **kwargs)
        frame = self._raw_frame_from_state(state.data)
        history = {
            key: jnp.repeat(value[None], self._feature_spec.num_history_steps, axis=0)
            for key, value in frame.items()
        }
        state = state.replace(
            info={
                **state.info,
                **{f"smp_{key}_history": value for key, value in history.items()},
                "smp_step": jnp.array(0, dtype=jnp.int32),
                "smp_rng": reward_rng,
            }
        )
        metrics = {
            **state.metrics,
            "smp/task_reward": state.reward,
            "smp/reward": jnp.array(0.0, dtype=state.reward.dtype),
            "smp/combined_reward": state.reward,
            "smp/sds_loss_mean": jnp.array(0.0, dtype=state.reward.dtype),
            "smp/sds_loss_std": jnp.array(0.0, dtype=state.reward.dtype),
        }
        return state.replace(metrics=metrics)

    def step(
        self,
        state: wrapper.mjx_env.State,
        action: jax.Array,
    ) -> wrapper.mjx_env.State:
        prev_history = {
            key: state.info[f"smp_{key}_history"]
            for key in (
                "root_pos",
                "root_quat",
                "root_vel",
                "root_ang_vel",
                "joints",
                "key_body_pos",
            )
        }
        prev_step = state.info["smp_step"]
        prev_rng = state.info["smp_rng"]

        next_state = self.env.step(state, action)
        frame = self._raw_frame_from_state(next_state.data)
        candidate_history = {
            key: jnp.roll(value, shift=-1, axis=0).at[-1].set(frame[key])
            for key, value in prev_history.items()
        }
        smp_step = prev_step + 1
        should_append = (smp_step % self._smp_stride) == 0
        history = {
            key: jnp.where(should_append, candidate_history[key], prev_history[key])
            for key in prev_history
        }
        rng, reward_rng = jax.random.split(prev_rng)

        next_state = next_state.replace(
            info={
                **next_state.info,
                **{f"smp_{key}_history": value for key, value in history.items()},
                "smp_step": smp_step,
                "smp_rng": rng,
            }
        )
        return self._replace_reward(next_state, reward_rng)

    def _raw_frame_from_state(self, data: mjx.Data) -> dict[str, jnp.ndarray]:
        key_body_pos = jnp.stack(
            [
                self.env._get_bodies_pos(data, flatten=False)[name]
                for name in self._feature_spec.key_body_names
            ],
            axis=0,
        )
        return {
            "root_pos": self.env.root_body(data).xpos,
            "root_quat": self.env.root_body(data).xquat,
            "root_vel": data.qvel[:3],
            "root_ang_vel": data.qvel[3:6],
            "joints": self.env._get_joint_angles(data),
            "key_body_pos": key_body_pos,
        }

    def _replace_reward(
        self,
        state: wrapper.mjx_env.State,
        reward_rng: jax.Array | None = None,
    ) -> wrapper.mjx_env.State:
        if reward_rng is None:
            reward_rng = state.info["smp_rng"]
        x_obs = compute_smp_obs(
            root_pos=state.info["smp_root_pos_history"],
            root_quat=state.info["smp_root_quat_history"],
            root_vel=state.info["smp_root_vel_history"],
            root_ang_vel=state.info["smp_root_ang_vel_history"],
            joints=state.info["smp_joints_history"],
            key_body_pos=state.info["smp_key_body_pos_history"],
        )[None]
        smp_reward, smp_metrics = compute_smp_reward(
            params=self._prior_params,
            normalizer=self._prior_normalizer,
            diff_normalizer=self._diff_normalizer,
            x_obs=x_obs,
            rng=reward_rng,
            model_config=self._model_config,
            reward_config=self._reward_config,
            model=self._prior_model,
        )
        task_reward = state.reward
        combined = blend_task_and_smp_rewards(
            task_reward=task_reward,
            smp_reward=smp_reward[0],
            reward_config=self._reward_config,
        )
        metrics = {
            **state.metrics,
            "smp/task_reward": task_reward,
            "smp/reward": smp_reward[0],
            "smp/combined_reward": combined,
            "smp/sds_loss_mean": smp_metrics["sds_loss_mean"],
            "smp/sds_loss_std": smp_metrics["sds_loss_std"],
        }
        return state.replace(reward=combined, metrics=metrics)

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def dt(self):
        return self.env.dt

    @property
    def unwrapped(self):
        return self

    @property
    def _mjx_model(self):
        return self.env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.env._mjx_model = value

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
