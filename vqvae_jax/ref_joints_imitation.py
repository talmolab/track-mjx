"""Reference-joints imitation target: raw joint angles for encoder input.

Instead of encoding the full 640-dim transformed reference (root, quat,
joints, bodies), this subclass adds a ``ref_joints`` key to the obs dict
containing only the raw reference joint angles (state-independent).

The observation dict gains a third key ``ref_joints`` alongside the
existing ``task_obs`` and ``proprioception``.  The normalizer
(``DictRunningStatisticsState``) only normalizes the two original keys;
``ref_joints`` passes through raw since joint angles are bounded.
"""

from typing import Any, Mapping

import jax.numpy as jp
from mujoco import mjx
from vnl_playground.tasks.rodent import imitation


class RefJointsImitation(imitation.Imitation):
    """Imitation env whose obs dict includes raw reference joint angles.

    Adds ``ref_joints`` (shape ``[T*n_joints]``) to the observation dict.
    ``task_obs`` remains the standard error-based signal from the
    parent class.
    """

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = super()._get_obs(data, info)
        reference = self._get_imitation_reference(data, info)
        obs["state"]["ref_joints"] = reference.joints.reshape(-1)
        return obs
