"""Reference-direct imitation target: state-independent encoder input.

Instead of encoding ``ref - current`` (state-dependent corrections), this
subclass computes imitation targets expressed entirely in the reference
trajectory's own egocentric frame.  Codes therefore encode *what kind of
motion the reference does*, independent of the agent's current state.

The observation dict shape and keys are identical to the parent class (same
470-dim ``imitation_target``, same ``proprioception``).  Only the *semantics*
of the values change.
"""

import collections
from typing import Any, Mapping

import brax.math
import jax
import jax.numpy as jp
from mujoco import mjx
from vnl_playground.tasks.rodent import imitation


class RefDirectImitation(imitation.Imitation):
    """Imitation env whose encoder input is reference-only (no agent state).

    Differences from the parent ``Imitation._get_imitation_target``:

    +-----------+----------------------------+-------------------------------+
    | Component | Parent (error-based)       | This (reference-direct)       |
    +-----------+----------------------------+-------------------------------+
    | root      | rotate(ref-cur, cur_quat)  | rotate(ref-ref[0], ref_quat0) |
    | quat      | rel_quat(ref, cur)         | rel_quat(ref, ref_quat0)      |
    | joint     | ref_joints - cur_joints    | ref_joints (raw)              |
    | body      | rotate(ref-cur, cur_quat)  | rotate(ref-ref0, ref_quat0)   |
    +-----------+----------------------------+-------------------------------+

    Frame 0 of root is always ``[0,0,0]`` and frame 0 of quat is always
    ``[1,0,0,0]`` (identity quaternion).
    """

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        reference = self._get_imitation_reference(data, info)

        # Reference frame-0 as the egocentric anchor
        ref_root0 = reference.root_position[0]  # [3]
        ref_quat0 = reference.root_quaternion[0]  # [4]

        # Root positions relative to frame-0, rotated into frame-0's frame
        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(ref_pos - ref_root0, ref_quat0)
        )(reference.root_position)

        # Quaternions relative to frame-0
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(ref_quat, ref_quat0)
        )(reference.root_quaternion)

        # Raw joint angles (no subtraction of current state)
        joint_targets = reference.joints

        # Body positions relative to frame-0 root, rotated into frame-0's frame
        bodies_pos = self._get_bodies_pos(data, flatten=False)
        body_ref_pos = jp.array(
            [reference.body_xpos(name) for name in bodies_pos]
        )  # [n_bodies, n_frames, 3]
        body_rel = body_ref_pos - ref_root0  # broadcast [n_bodies, n_frames, 3]
        to_ref_ego = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, ref_quat0))
        body_targets = jax.vmap(to_ref_ego)(body_rel)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )
