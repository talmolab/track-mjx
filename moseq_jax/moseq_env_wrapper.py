"""Imitation environment subclass that includes pre-computed KPMS codes.

Subclasses ``Imitation`` to inject ``kpms_code`` directly in ``_get_obs``,
and flattens the v0.0.13 nested obs structure to a flat dict. This avoids
the pitfalls of an env *wrapper* approach where:

* ``BraxDomainRandomizationVmapWrapper`` bypasses wrappers via ``env.unwrapped``
* ``EpisodeWrapper.step``'s ``jax.lax.scan`` requires carry-in / carry-out to
  have identical pytree structure, but the base env's ``_get_obs`` would strip
  the extra key.

By producing the flat ``{"task_obs": ..., "proprioception": ..., "kpms_code": ...}``
inside ``_get_obs``, every wrapper (Vmap, DR, Episode, AutoReset) sees a consistent
obs structure at all times.
"""

import collections
from typing import Any, Mapping

import jax
import jax.numpy as jp
import jax.numpy as jnp
from mujoco import mjx
from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips


class MoSeqImitation(Imitation):
    """Imitation env whose observations include pre-computed KPMS syllable codes.

    Overrides ``_get_obs`` to:
    1. Inject ``kpms_code`` into the obs dict.
    2. Flatten nested obs values to 1D arrays.
    3. Strip the ``state`` hierarchy (v0.0.13 nesting).

    This produces a flat dict ``{"task_obs": 1D, "proprioception": 1D, "kpms_code": 1D}``
    at all times, avoiding wrapper-chain pitfalls.

    Attributes:
        _kpms_codes: Pre-computed code array ``[n_clips, n_frames]``.
        _code_stack_size: Number of consecutive codes to stack (1 = current only,
            5 = current + 4 future codes).
    """

    def __init__(
        self,
        config: Any,
        clips: ReferenceClips,
        kpms_codes: jnp.ndarray,
        code_stack_size: int = 1,
    ):
        super().__init__(config=config, clips=clips)
        self._kpms_codes = jnp.asarray(kpms_codes, dtype=jnp.int32)
        self._code_stack_size = code_stack_size

    # ------------------------------------------------------------------
    # Override _get_obs to inject kpms_code and flatten obs
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = super()._get_obs(data, info)
        frame = self._get_cur_frame(data, info)
        n_frames = self._kpms_codes.shape[1]
        clip_idx = info["reference_clip"]

        # Inject kpms_code
        if self._code_stack_size <= 1:
            frame = jnp.clip(frame, 0, n_frames - 1)
            code = self._kpms_codes[clip_idx, frame]
            kpms_code = code[..., None].astype(jnp.float32)
        else:
            offsets = jnp.arange(self._code_stack_size)
            frames = jnp.clip(frame[..., None] + offsets, 0, n_frames - 1)
            codes = self._kpms_codes[clip_idx[..., None], frames]
            kpms_code = codes.astype(jnp.float32)

        # Flatten nested obs and strip "state" hierarchy:
        # v0.0.13 returns {"state": OrderedDict(task_obs=nested, proprioception=nested)}
        # We produce {"task_obs": 1D, "proprioception": 1D, "kpms_code": 1D}
        state_obs = obs["state"]
        flat = collections.OrderedDict()
        for k, v in state_obs.items():
            flat[k] = jnp.nan_to_num(jax.flatten_util.ravel_pytree(v)[0])
        flat["kpms_code"] = kpms_code
        return flat

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            jax.flatten_util.ravel_pytree(obs_size["proprioception"])[0]
        )

    @property
    def observation_size(self):
        obs = self.non_flattened_observation_size
        return jp.sum(jax.flatten_util.ravel_pytree(obs)[0])
