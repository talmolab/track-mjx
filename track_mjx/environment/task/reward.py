"""All reward calculation and reward calculation helper functions"""

import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

from jax.numpy import inf, ndarray
import mujoco
from mujoco import mjx

import numpy as np

import os


def _bounded_quat_dist(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Computes a quaternion distance limiting the difference to a max of pi/2.

    This function supports an arbitrary number of batch dimensions, B.

    Args:
        source: a quaternion, shape (B, 4).
        target: another quaternion, shape (B, 4).

    Returns:
        Quaternion distance, shape (B, 1).
    """
    source /= jp.linalg.norm(source, axis=-1, keepdims=True)
    target /= jp.linalg.norm(target, axis=-1, keepdims=True)
    # "Distance" in interval [-1, 1].
    dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
    # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
    dist = jp.minimum(1.0, dist)
    # Divide by 2 and add an axis to ensure consistency with expected return
    # shape and magnitude.
    return 0.5 * jp.arccos(dist)[..., np.newaxis]


#TODO: using Brax's style, should we keep all the reward separated?
def compute_tracking_rewards(
    data,
    reference_clip,
    walker,
    action,
    info,
    healthy_z_range,
    too_far_dist,
    bad_pose_dist,
    bad_quat_dist,
    pos_reward_weight: float = 1.0,
    quat_reward_weight: float = 1.0,
    joint_reward_weight: float = 1.0,
    angvel_reward_weight: float = 1.0,
    bodypos_reward_weight: float = 1.0,
    endeff_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 1.0,
    ctrl_diff_cost_weight: float = 1.0,
) -> tuple[float, dict]:
    """
    Computes the tracking reward for motion imitation
    """

    pos_distance = data.qpos[:3] - reference_clip.position
    pos_reward = pos_reward_weight * jp.exp(-400 * jp.sum(pos_distance**2))

    quat_distance = jp.sum(
        _bounded_quat_dist(data.qpos[3:7], reference_clip.quaternion) ** 2
    )
    quat_reward = quat_reward_weight * jp.exp(-4.0 * quat_distance)

    joint_distance = jp.sum((data.qpos[7:] - reference_clip.joints) ** 2)
    joint_reward = joint_reward_weight * jp.exp(-0.25 * joint_distance)
    info["joint_distance"] = joint_distance

    angvel_reward = angvel_reward_weight * jp.exp(
        -0.5 * jp.sum((data.qvel[3:6] - reference_clip.angular_velocity) ** 2)
    )

    bodypos_reward = bodypos_reward_weight * jp.exp(
        -8.0
        * jp.sum(
            (
                walker.get_body_positions(data.xpos)
                - reference_clip.body_positions[walker._body_idxs]
            ).flatten()
            ** 2
        )
    )

    endeff_reward = endeff_reward_weight * jp.exp(
        -500
        * jp.sum(
            (
                walker.get_end_effector_positions(data.xpos)
                - reference_clip.body_positions[walker._endeff_idxs]
            ).flatten()
            ** 2
        )
    )

    min_z, max_z = healthy_z_range
    is_healthy = jp.where(walker.get_torso_position(data.xpos)[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(walker.get_torso_position(data.xpos)[2] > max_z, 0.0, is_healthy)
    fall = 1.0 - is_healthy

    summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
    too_far = jp.where(summed_pos_distance > too_far_dist, 1.0, 0.0)
    info["summed_pos_distance"] = summed_pos_distance
    info["quat_distance"] = quat_distance
    bad_pose = jp.where(joint_distance > bad_pose_dist, 1.0, 0.0)
    bad_quat = jp.where(quat_distance > bad_quat_dist, 1.0, 0.0)
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    ctrl_diff_cost = ctrl_diff_cost_weight * jp.sum(
        jp.square(info["prev_ctrl"] - action)
    )

    return (
        pos_reward,
        quat_reward,
        joint_reward,
        angvel_reward,
        bodypos_reward,
        endeff_reward,
        ctrl_cost,
        ctrl_diff_cost,
        too_far,
        bad_pose,
        bad_quat,
        fall,
        info,
    )
