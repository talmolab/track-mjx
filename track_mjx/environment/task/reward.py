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


def compute_pos_reward(data, reference_clip, weight):
    """Position based reward"""
    pos_distance = data.qpos[:3] - reference_clip.position
    return weight * jp.exp(-400 * jp.sum(pos_distance**2)), pos_distance


def compute_quat_reward(data, reference_clip, weight):
    """quaternion based reward"""
    quat_distance = jp.sum(
        _bounded_quat_dist(data.qpos[3:7], reference_clip.quaternion) ** 2
    )
    return weight * jp.exp(-4.0 * quat_distance), quat_distance


def compute_joint_reward(data, reference_clip, weight):
    """joint based reward"""
    joint_distance = jp.sum((data.qpos[7:] - reference_clip.joints) ** 2)
    return weight * jp.exp(-0.25 * joint_distance), joint_distance


def compute_angvel_reward(data, reference_clip, weight):
    """'angle velocity based reward"""
    return weight * jp.exp(
        -0.5 * jp.sum((data.qvel[3:6] - reference_clip.angular_velocity) ** 2)
    )


def compute_bodypos_reward(data, reference_clip, walker, weight):
    """body position based reward"""
    return weight * jp.exp(
        -8.0
        * jp.sum(
            (
                walker.get_body_positions(data.xpos)
                - reference_clip.body_positions[walker.body_idxs]
            ).flatten()
            ** 2
        )
    )


def compute_endeff_reward(data, reference_clip, walker, weight):
    """end effector based reward"""
    return weight * jp.exp(
        -500
        * jp.sum(
            (
                walker.get_end_effector_positions(data.xpos)
                - reference_clip.body_positions[walker.endeff_idxs]
            ).flatten()
            ** 2
        )
    )


def compute_ctrl_cost(action, weight):
    """control cost of action"""
    return weight * jp.sum(jp.square(action))


def compute_ctrl_diff_cost(action, prev_action, weight):
    """control cost in differences in actions"""
    return weight * jp.sum(jp.square(prev_action - action))


def compute_health_penalty(data, walker, healthy_z_range):
    """healthy z range reward"""
    min_z, max_z = healthy_z_range
    torso_z = walker.get_torso_position(data.xpos)[2]
    is_healthy = jp.where(torso_z < min_z, 0.0, 1.0)
    is_healthy = jp.where(torso_z > max_z, 0.0, is_healthy)
    fall = 1.0 - is_healthy
    return fall


def compute_penalty_terms(
    pos_distance,
    joint_distance,
    quat_distance,
    too_far_dist,
    bad_pose_dist,
    bad_quat_dist,
):
    """all distance based penalty terms"""
    summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
    too_far = jp.where(summed_pos_distance > too_far_dist, 1.0, 0.0)
    bad_pose = jp.where(joint_distance > bad_pose_dist, 1.0, 0.0)
    bad_quat = jp.where(quat_distance > bad_quat_dist, 1.0, 0.0)
    return too_far, bad_pose, bad_quat, summed_pos_distance


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
    Computes all the tracking reward for motion imitation
    """

    pos_reward, pos_distance = compute_pos_reward(
        data, reference_clip, pos_reward_weight
    )
    quat_reward, quat_distance = compute_quat_reward(
        data, reference_clip, quat_reward_weight
    )
    joint_reward, joint_distance = compute_joint_reward(
        data, reference_clip, joint_reward_weight
    )
    angvel_reward = compute_angvel_reward(data, reference_clip, angvel_reward_weight)
    bodypos_reward = compute_bodypos_reward(
        data, reference_clip, walker, bodypos_reward_weight
    )
    endeff_reward = compute_endeff_reward(
        data, reference_clip, walker, endeff_reward_weight
    )
    ctrl_cost = compute_ctrl_cost(action, ctrl_cost_weight)
    ctrl_diff_cost = compute_ctrl_diff_cost(
        action, info["prev_ctrl"], ctrl_diff_cost_weight
    )
    fall = compute_health_penalty(data, walker, healthy_z_range)
    too_far, bad_pose, bad_quat, summed_pos_distance = compute_penalty_terms(
        pos_distance,
        joint_distance,
        quat_distance,
        too_far_dist,
        bad_pose_dist,
        bad_quat_dist,
    )

    info["joint_distance"] = joint_distance
    info["summed_pos_distance"] = summed_pos_distance
    info["quat_distance"] = quat_distance

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
