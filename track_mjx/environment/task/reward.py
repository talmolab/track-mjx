"""All reward calculation and reward calculation helper functions"""

import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

from jax import numpy as jp
import mujoco
from mujoco import mjx

import numpy as np

import os

from track_mjx.environment.walker.base import BaseWalker
from track_mjx.io.preprocess.mjx_preprocess import ReferenceClip
from mujoco import MjData

POS_REWARD_EXP_SCALE = 400.0
QUAT_REWARD_EXP_SCALE = 4.0
JOINT_REWARD_EXP_SCALE = 0.25
ANGVEL_REWARD_EXP_SCALE = 0.5
BODYPOS_REWARD_EXP_SCALE = 8.0
ENDEFF_REWARD_EXP_SCALE = 500.0
CTRL_COST_EXP_SCALE = 1.0
CTRL_DIFF_COST_EXP_SCALE = 1.0
PENALTY_POS_DISTANCE_SCALE = jp.array([1.0, 1.0, 0.2])


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


def compute_pos_reward(
    pos_array: jp.ndarray, reference_clip_pos: jp.ndarray, weight: float
) -> tuple[jp.ndarray, jp.ndarray]:
    """Position-based reward.

    Args:
        pos_array: Current position data.
        reference_clip_pos: Reference trajectory position data.
        weight: Weight for the reward.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Weighted position reward and position distance.
    """
    pos_distance = pos_array - reference_clip_pos
    weighted_pos_reward = weight * jp.exp(
        -POS_REWARD_EXP_SCALE * jp.sum(pos_distance**2)
    )
    return weighted_pos_reward, pos_distance


def compute_quat_reward(
    quat_array: jp.ndarray, reference_clip_quat: jp.ndarray, weight: float
) -> tuple[jp.ndarray, jp.ndarray]:
    """Quaternion-based reward.

    Args:
        quat_array: Current quaternion data.
        reference_clip_quat: Reference trajectory data.
        weight: Weight for the reward.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Weighted quaternion reward and quaternion distance.
    """
    quat_distance = jp.sum(_bounded_quat_dist(quat_array, reference_clip_quat) ** 2)
    weighted_quat_reward = weight * jp.exp(-QUAT_REWARD_EXP_SCALE * quat_distance)
    return weighted_quat_reward, quat_distance


def compute_joint_reward(
    joint_array: jp.ndarray, reference_clip_joint: jp.ndarray, weight: float
) -> tuple[jp.ndarray, jp.ndarray]:
    """Joint-based reward.

    Args:
        joint_array: Current joint position data.
        reference_clip_joint: Reference trajectory joint position data.
        weight: Weight for the reward.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Weighted joint reward and joint distance.
    """
    joint_distance = jp.sum((joint_array - reference_clip_joint) ** 2)
    weighted_joint_reward = weight * jp.exp(-JOINT_REWARD_EXP_SCALE * joint_distance)
    return weighted_joint_reward, joint_distance


def compute_angvel_reward(
    angvel_array: jp.ndarray, reference_clip_angvel: jp.ndarray, weight: float
) -> jp.ndarray:
    """Angular velocity-based reward.

    Args:
        angvel_array: Current angular velocity data.
        reference_clip_angvel: Reference trajectory angular velocity data.
        weight: Weight for the reward.

    Returns:
        jp.ndarray: Weighted angular velocity reward.
    """
    weighted_angvel_reward = weight * jp.exp(
        -ANGVEL_REWARD_EXP_SCALE * jp.sum((angvel_array - reference_clip_angvel) ** 2)
    )
    return weighted_angvel_reward


def compute_bodypos_reward(
    bodypos_array: jp.ndarray, reference_clip_bodypos: jp.ndarray, weight: float
) -> jp.ndarray:
    """Body position-based reward.

    Args:
        bodypos_array: Current body position data.
        reference_clip: Reference trajectory body position data.
        weight: Weight for the reward.

    Returns:
        jp.ndarray: Weighted body position reward.
    """
    weighted_bodypos_reward = weight * jp.exp(
        -BODYPOS_REWARD_EXP_SCALE
        * jp.sum((bodypos_array - reference_clip_bodypos).flatten() ** 2)
    )
    return weighted_bodypos_reward


def compute_endeff_reward(
    endeff_array: jp.ndarray, reference_clip_endeff: jp.ndarray, weight: float
) -> jp.ndarray:
    """End-effector-based reward.

    Args:
        endeff_array: Current end-effector position data.
        reference_clip_endeff: Reference trajectory end-effector position data.
        weight: Weight for the reward.

    Returns:
        jp.ndarray: Weighted end-effector reward.
    """
    weighted_endeff_reward = weight * jp.exp(
        -ENDEFF_REWARD_EXP_SCALE
        * jp.sum((endeff_array - reference_clip_endeff).flatten() ** 2)
    )
    return weighted_endeff_reward


def compute_ctrl_cost(action: jp.ndarray, weight: float) -> jp.ndarray:
    """Control cost of action.

    Args:
        action: Action array.
        weight: Weight for the control cost.

    Returns:
        jp.ndarray: Weighted control cost.
    """
    weighted_ctrl_cost = weight * jp.sum(jp.square(action))
    return weighted_ctrl_cost


def compute_ctrl_diff_cost(
    action: jp.ndarray, prev_action: jp.ndarray, weight: float
) -> jp.ndarray:
    """Control cost for differences in actions.

    Args:
        action: Current action array.
        prev_action: Previous action array.
        weight: Weight for the control difference cost.

    Returns:
        jp.ndarray: Weighted control difference cost.
    """
    weighted_ctrl_diff_cost = weight * jp.sum(jp.square(prev_action - action))
    return weighted_ctrl_diff_cost


def compute_health_penalty(
    xpos_array: jp.ndarray, walker: BaseWalker, healthy_z_range: tuple[float, float]
) -> jp.ndarray:
    """Computes a penalty for being outside the healthy z-range.

    Args:
        xpos: Body positions array.
        walker: Walker based object.
        healthy_z_range: Minimum and maximum healthy z-range.

    Returns:
        jp.ndarray: Fall penalty (0 for healthy, 1 for unhealthy).
    """
    min_z, max_z = healthy_z_range
    torso_z = walker.get_torso_position(xpos_array)[2]
    is_healthy = jp.where(torso_z < min_z, 0.0, 1.0)
    is_healthy = jp.where(torso_z > max_z, 0.0, is_healthy)
    fall = 1.0 - is_healthy
    return fall


def compute_penalty_terms(
    pos_distance: jp.ndarray,
    joint_distance: jp.ndarray,
    quat_distance: jp.ndarray,
    too_far_dist: float,
    bad_pose_dist: float,
    bad_quat_dist: float,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """Computes penalty terms based on distances.

    Args:
        pos_distance: Distance in position space.
        joint_distance: Distance in joint space.
        quat_distance: Quaternion distance.
        too_far_dist: Threshold for position distance penalty.
        bad_pose_dist: Threshold for joint distance penalty.
        bad_quat_dist: Threshold for quaternion distance penalty.

    Returns:
        Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
            Penalties for being too far, bad pose, bad quaternion,
            and the summed position distance.
    """
    summed_pos_distance = jp.sum((pos_distance * PENALTY_POS_DISTANCE_SCALE) ** 2)
    too_far = jp.where(summed_pos_distance > too_far_dist, 1.0, 0.0)
    bad_pose = jp.where(joint_distance > bad_pose_dist, 1.0, 0.0)
    bad_quat = jp.where(quat_distance > bad_quat_dist, 1.0, 0.0)
    return too_far, bad_pose, bad_quat, summed_pos_distance


def compute_tracking_rewards(
    data: MjData,
    reference_clip: ReferenceClip,
    walker: BaseWalker,
    action: jp.ndarray,
    info: dict[str, jp.ndarray],
    healthy_z_range: tuple[float, float],
    too_far_dist: float,
    bad_pose_dist: float,
    bad_quat_dist: float,
    pos_reward_weight: float = 1.0,
    quat_reward_weight: float = 1.0,
    joint_reward_weight: float = 1.0,
    angvel_reward_weight: float = 1.0,
    bodypos_reward_weight: float = 1.0,
    endeff_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 1.0,
    ctrl_diff_cost_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Computes tracking rewards and penalties for motion imitation.

    Args:
        data: Current state MjData object.
        reference_clip: Reference trajectory objct.
        walker: Base walker object.
        action: Current action.
        info: Dictionary of information for logging
        healthy_z_range: Healthy z-range bounds.
        too_far_dist: Threshold for 'too far' penalty.
        bad_pose_dist: Threshold for 'bad pose' penalty.
        bad_quat_dist: Threshold for 'bad quaternion' penalty.
        pos_reward_weight: Weight for position reward.
        quat_reward_weight: Weight for quaternion reward.
        joint_reward_weight: Weight for joint reward.
        angvel_reward_weight: Weight for angular velocity reward.
        bodypos_reward_weight: Weight for body position reward.
        endeff_reward_weight: Weight for end-effector reward.
        ctrl_cost_weight: Weight for control cost.
        ctrl_diff_cost_weight: Weight for control difference cost.

    Returns:
        Tuple[float, Dict[str, float]]: Total reward and detailed info dictionary.
    """

    pos_array = data.qpos[:3]
    reference_clip_pos = reference_clip.position
    pos_reward, pos_distance = compute_pos_reward(
        pos_array, reference_clip_pos, pos_reward_weight
    )

    quat_array = data.qpos[3:7]
    reference_clip_quat = reference_clip.quaternion
    quat_reward, quat_distance = compute_quat_reward(
        quat_array, reference_clip_quat, quat_reward_weight
    )

    joint_array = data.qpos[7:]
    reference_clip_joint = reference_clip.joints
    joint_reward, joint_distance = compute_joint_reward(
        joint_array, reference_clip_joint, joint_reward_weight
    )

    angvel_array = data.qvel[3:6]
    reference_clip_angvel = reference_clip.angular_velocity
    angvel_reward = compute_angvel_reward(
        angvel_array, reference_clip_angvel, angvel_reward_weight
    )

    bodypos_array = walker.get_body_positions(data.xpos)
    reference_clip_bodypos = reference_clip.body_positions[walker.body_idxs]
    bodypos_reward = compute_bodypos_reward(
        bodypos_array, reference_clip_bodypos, walker, bodypos_reward_weight
    )

    endeff_array = walker.get_end_effector_positions(data.xpos)
    reference_clip_endeff = reference_clip.body_positions[walker.endeff_idxs]
    endeff_reward = compute_endeff_reward(
        endeff_array, reference_clip_endeff, walker, endeff_reward_weight
    )

    ctrl_cost = compute_ctrl_cost(action, ctrl_cost_weight)
    ctrl_diff_cost = compute_ctrl_diff_cost(
        action, info["prev_ctrl"], ctrl_diff_cost_weight
    )

    xpos_array = data.xpos
    fall = compute_health_penalty(xpos_array, walker, healthy_z_range)
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
