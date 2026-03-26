"""Imitation environment subclass that includes pre-computed KPMS codes.

Subclasses ``Imitation`` to inject ``kpms_code`` directly in ``_get_obs``,
so the obs pytree structure is consistent from the very start.  This avoids
the pitfalls of an env *wrapper* approach where:

* ``BraxDomainRandomizationVmapWrapper`` bypasses wrappers via ``env.unwrapped``
* ``EpisodeWrapper.step``'s ``jax.lax.scan`` requires carry-in / carry-out to
  have identical pytree structure, but the base env's ``_get_obs`` would strip
  the extra key.

By producing ``kpms_code`` inside ``_get_obs``, every wrapper (Vmap, DR,
Episode, AutoReset) sees a consistent 3-key ``OrderedDict`` at all times.
"""

from typing import Any, Mapping

import jax.numpy as jnp
from mujoco import mjx
from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips


class MoSeqImitation(Imitation):
    """Imitation env whose observations include pre-computed KPMS syllable codes.

    Attributes:
        _kpms_codes: Pre-computed code array ``[n_clips, n_frames]``.
    """

    def __init__(
        self,
        config: Any,
        clips: ReferenceClips,
        kpms_codes: jnp.ndarray,
    ):
        super().__init__(config=config, clips=clips)
        self._kpms_codes = jnp.asarray(kpms_codes, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Override _get_obs to append kpms_code
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = super()._get_obs(data, info)
        frame = self._get_cur_frame(data, info)
        frame = jnp.clip(frame, 0, self._kpms_codes.shape[1] - 1)
        code = self._kpms_codes[info["reference_clip"], frame]
        # Inject kpms_code into the "state" sub-dict (vnl-playground now nests
        # obs under {"state": {"task_obs": ..., "proprioception": ...}})
        obs["state"]["kpms_code"] = code[..., None].astype(jnp.float32)
        return obs
