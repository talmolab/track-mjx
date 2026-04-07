"""Environment wrappers for AMP training."""

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
from mujoco_playground import wrapper

from track_mjx.agent.amp_ppo import features


_INFO_PREFIX = "amp_"
_HISTORY_KEYS = (
    "root_pos",
    "root_quat",
    "root_vel",
    "root_ang_vel",
    "joints",
    "joint_vels",
    "key_pos",
)


class AMPObsWrapper(wrapper.Wrapper):
    """Adds AMP discriminator observations to ``state.info``.

    The policy observation is left unchanged. Generated motion histories are
    stored in ``state.info`` so Brax rollouts can collect ``amp_obs`` through
    ``extra_fields``.
    """

    def __init__(
        self,
        env: Any,
        num_disc_obs_steps: int = 10,
        key_body_names: Sequence[str] = (),
    ):
        super().__init__(env)
        self._num_disc_obs_steps = int(num_disc_obs_steps)
        self._key_body_names = tuple(key_body_names)

    def reset(self, rng, **kwargs):
        state = self.env.reset(rng, **kwargs)
        histories = self._reference_histories(
            state.info["reference_clip"],
            state.info["start_frame"],
        )
        return self._replace_amp_info(state, histories)

    def step(self, state, action):
        next_state = self.env.step(state, action)
        histories = self._append_current(next_state.info, next_state.data)
        return self._replace_amp_info(next_state, histories)

    @property
    def _mjx_model(self):
        return self.env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.env._mjx_model = value

    def _reference_histories(self, clip_id, latest_frame) -> Mapping[str, jnp.ndarray]:
        offsets = self._num_disc_obs_steps - 1 - jnp.arange(self._num_disc_obs_steps)
        frame_ids = jnp.clip(
            latest_frame - offsets,
            0,
            self.env.reference_clips.qpos.shape[1] - 1,
        )
        return features.reference_histories(
            self.env.reference_clips,
            clip_id,
            frame_ids,
            self._key_body_names,
        )

    def _append_current(
        self, info: Mapping[str, Any], data
    ) -> Mapping[str, jnp.ndarray]:
        current = self._current_components(data)
        return {
            key: jnp.concatenate(
                [info[f"{_INFO_PREFIX}{key}_hist"][1:], current[key][None, ...]],
                axis=0,
            )
            for key in _HISTORY_KEYS
        }

    def _current_components(self, data) -> Mapping[str, jnp.ndarray]:
        root = self.env.root_body(data)
        if self._key_body_names:
            body_pos = self.env._get_bodies_pos(data, flatten=False)
            key_pos = jnp.stack([body_pos[name] for name in self._key_body_names])
        else:
            key_pos = jnp.zeros((0, 3), dtype=data.qpos.dtype)

        return {
            "root_pos": root.xpos,
            "root_quat": root.xquat,
            "root_vel": data.qvel[:3],
            "root_ang_vel": data.qvel[3:6],
            "joints": self.env._get_joint_angles(data),
            "joint_vels": self.env._get_joint_ang_vels(data),
            "key_pos": key_pos,
        }

    def _replace_amp_info(self, state, histories: Mapping[str, jnp.ndarray]):
        info = dict(state.info)
        for key, value in histories.items():
            info[f"{_INFO_PREFIX}{key}_hist"] = value
        info["amp_obs"] = features.compute_amp_obs(**histories)
        return state.replace(info=info)
