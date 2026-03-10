"""Environment wrapper that injects pre-computed KPMS codes into observations.

Wraps an imitation environment to add a ``kpms_code`` key to the observation
dictionary at each step.  The code is looked up from a pre-computed table
indexed by (reference_clip, current_frame).
"""

from typing import Any

import jax.numpy as jnp


class MoSeqCodeWrapper:
    """Injects pre-computed KPMS syllable codes into the observation dict.

    Attributes:
        env: Wrapped environment instance.
    """

    def __init__(self, env: Any, kpms_codes: jnp.ndarray):
        """Initialize the wrapper.

        Args:
            env: Base imitation environment.
            kpms_codes: Pre-computed code array, shape ``[n_clips, n_frames]``
                with integer syllable labels.
        """
        self.env = env
        self._kpms_codes = jnp.asarray(kpms_codes, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Core env methods
    # ------------------------------------------------------------------

    def _lookup_code(self, data: Any, info: dict) -> jnp.ndarray:
        """Look up the KPMS code for the current (clip, frame)."""
        frame = self.env.unwrapped._get_cur_frame(data, info)
        frame = jnp.clip(frame, 0, self._kpms_codes.shape[1] - 1)
        return self._kpms_codes[info["reference_clip"], frame]

    def _inject_code(self, obs: dict, code: jnp.ndarray) -> dict:
        """Add ``kpms_code`` (shape ``[..., 1]``, float32) to *obs*."""
        return {**obs, "kpms_code": code[..., None].astype(jnp.float32)}

    def reset(self, rng, **kwargs):
        state = self.env.reset(rng, **kwargs)
        code = self._lookup_code(state.data, state.info)
        return state.replace(obs=self._inject_code(state.obs, code))

    def step(self, state, action):
        state = self.env.step(state, action)
        code = self._lookup_code(state.data, state.info)
        return state.replace(obs=self._inject_code(state.obs, code))

    # ------------------------------------------------------------------
    # Forwarded properties / methods
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def proprioceptive_obs_size(self):
        return self.env.proprioceptive_obs_size

    def __getattr__(self, name: str):
        return getattr(self.env, name)
