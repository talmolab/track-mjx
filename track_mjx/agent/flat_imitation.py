"""Imitation subclass that flattens v0.0.13 nested obs inline.

vnl-playground v0.0.13 ``Imitation._get_obs`` returns nested obs::

    {"state": OrderedDict(task_obs=nested, proprioception=nested)}

``BraxDomainRandomizationVmapWrapper`` bypasses wrapper chains via
``env.unwrapped``, so ``LegacyObsWrapper(TrackMjxObsWrapper(...))``
is never applied during DR reset/step. This subclass flattens obs
directly in ``_get_obs`` so the flat structure is visible to ALL
wrappers at all times.
"""

import collections
from typing import Any, Mapping

import jax
import jax.numpy as jp
from mujoco import mjx
from vnl_playground.tasks.rodent.imitation import Imitation


class FlatImitation(Imitation):
    """Imitation env that produces flat obs dict without nesting.

    Overrides ``_get_obs`` to flatten nested obs values to 1D arrays
    and strip the ``state`` hierarchy, producing::

        {"task_obs": 1D_array, "proprioception": 1D_array}
    """

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = super()._get_obs(data, info)
        state_obs = obs["state"]
        return collections.OrderedDict(
            (k, jp.nan_to_num(jax.flatten_util.ravel_pytree(v)[0]))
            for k, v in state_obs.items()
        )

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
