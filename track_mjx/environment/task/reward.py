"""All reward calculation and reward calculation helper functions"""

from jax import numpy as jp
import jax
import numpy as np
from typing import Union
from omegaconf import ListConfig
from track_mjx.environment.walker.base import BaseWalker
from track_mjx.io.load import ReferenceClip
import mujoco

from flax import struct


@struct.dataclass
class RewardConfig:
    """Weights and scales for the imitation reward terms.
    Initialized through env_config.reward_weights in the config.yaml file.
    """

    too_far_dist: float
    bad_pose_dist: float
    bad_quat_dist: float
    ctrl_cost_weight: float
    ctrl_diff_cost_weight: float
    energy_cost_weight: float
    energy_cost_weight: float
    pos_reward_weight: float
    quat_reward_weight: float
    joint_reward_weight: float
    angvel_reward_weight: float
    bodypos_reward_weight: float
    endeff_reward_weight: float
    healthy_z_range: tuple[float, float]
    pos_reward_exp_scale: float
    quat_reward_exp_scale: float
    joint_reward_exp_scale: float
    angvel_reward_exp_scale: float
    bodypos_reward_exp_scale: float
    endeff_reward_exp_scale: float
    penalty_pos_distance_scale: jp.ndarray
    var_window_size: int = 50
    var_coeff: float = 5e-2
    jerk_coeff: float = 5e-4

    def __post_init__(self):
        if isinstance(self.penalty_pos_distance_scale, list) or isinstance(
            self.penalty_pos_distance_scale, ListConfig
        ):
            object.__setattr__(
                self,
                "penalty_pos_distance_scale",
                jp.array(self.penalty_pos_distance_scale),
            )


def _bounded_quat_dist(source: jp.ndarray, target: jp.ndarray) -> jp.ndarray:
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
    pos_array: jp.ndarray,
    reference_clip_pos: jp.ndarray,
    weight: float,
    pos_reward_exp_scale: float,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Position-based reward.

    Args:
        pos_array: Current position data.
        reference_clip_pos: Reference trajectory position data.
        weight: Weight for the reward.
        pos_reward_exp_scale: Scaling factor for position rewards.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Weighted position reward and position distance.
    """
    pos_distance = pos_array - reference_clip_pos
    weighted_pos_reward = weight * jp.exp(
        -pos_reward_exp_scale * jp.sum(pos_distance**2)
    )
    return weighted_pos_reward, pos_distance


def compute_quat_reward(
    quat_array: jp.ndarray,
    reference_clip_quat: jp.ndarray,
    weight: float,
    quat_reward_exp_scale: float,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Quaternion-based reward.

    Args:
        quat_array: Current quaternion data.
        reference_clip_quat: Reference trajectory data.
        weight: Weight for the reward.
        quat_reward_exp_scale: Scaling factor for quaternion rewards.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Weighted quaternion reward and quaternion distance.
    """
    quat_distance = jp.sum(_bounded_quat_dist(quat_array, reference_clip_quat) ** 2)
    weighted_quat_reward = weight * jp.exp(-quat_reward_exp_scale * quat_distance)
    return weighted_quat_reward, quat_distance


def compute_joint_reward(
    joint_array: jp.ndarray,
    reference_clip_joint: jp.ndarray,
    weight: float,
    joint_reward_exp_scale: float,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Joint-based reward.

    Args:
        joint_array: Current joint position data.
        reference_clip_joint: Reference trajectory joint position data.
        weight: Weight for the reward.
        joint_reward_exp_scale: Scaling factor for joint rewards.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Weighted joint reward and joint distance.
    """
    joint_distance = jp.sum((joint_array - reference_clip_joint) ** 2)
    weighted_joint_reward = weight * jp.exp(-joint_reward_exp_scale * joint_distance)
    return weighted_joint_reward, joint_distance


def compute_angvel_reward(
    angvel_array: jp.ndarray,
    reference_clip_angvel: jp.ndarray,
    weight: float,
    angvel_reward_exp_scale: float,
) -> jp.ndarray:
    """Angular velocity-based reward.

    Args:
        angvel_array: Current angular velocity data.
        reference_clip_angvel: Reference trajectory angular velocity data.
        weight: Weight for the reward.
        angvel_reward_exp_scale: Scaling factor for angular velocity rewards.

    Returns:
        jp.ndarray: Weighted angular velocity reward.
    """
    weighted_angvel_reward = weight * jp.exp(
        -angvel_reward_exp_scale * jp.sum((angvel_array - reference_clip_angvel) ** 2)
    )
    return weighted_angvel_reward


def compute_bodypos_reward(
    bodypos_array: jp.ndarray,
    reference_clip_bodypos: jp.ndarray,
    weight: float,
    bodypos_reward_exp_scale: float,
) -> jp.ndarray:
    """Body position-based reward.

    Args:
        bodypos_array: Current body position data.
        reference_clip: Reference trajectory body position data.
        weight: Weight for the reward.
        bodypos_reward_exp_scale: Scaling factor for body position rewards.

    Returns:
        jp.ndarray: Weighted body position reward.
    """
    weighted_bodypos_reward = weight * jp.exp(
        -bodypos_reward_exp_scale
        * jp.sum((bodypos_array - reference_clip_bodypos).flatten() ** 2)
    )
    return weighted_bodypos_reward


def compute_endeff_reward(
    endeff_array: jp.ndarray,
    reference_clip_endeff: jp.ndarray,
    weight: float,
    endeff_reward_exp_scale: float,
) -> jp.ndarray:
    """End-effector-based reward.

    Args:
        endeff_array: Current end-effector position data.
        reference_clip_endeff: Reference trajectory end-effector position data.
        weight: Weight for the reward.
        endeff_reward_exp_scale: Scaling factor for end-effector rewards.

    Returns:
        jp.ndarray: Weighted end-effector reward.
    """
    weighted_endeff_reward = weight * jp.exp(
        -endeff_reward_exp_scale
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


def compute_energy_cost(
    qvel: jp.ndarray, qfrc_actuator: jp.ndarray, weight: float
) -> jp.ndarray:
    """Penalize energy consumption.
    Args:
        qvel: Velocity data of joints.
        qfrc_actuator: Actuator force data.
    Returns:
        jp.ndarray: Weighted energy cost.
    """
    return weight * jp.minimum(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)), 50.0)


def compute_health_penalty(
    torso_z: jp.ndarray, healthy_z_range: tuple[float, float]
) -> jp.ndarray:
    """Computes a penalty for being outside the healthy z-range.

    Args:
        torso_z: Torso z-position.
        healthy_z_range: Minimum and maximum healthy z-range.

    Returns:
        jp.ndarray: Fall penalty (0 for healthy, 1 for unhealthy).
    """
    min_z, max_z = healthy_z_range
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
    penalty_pos_distance_scale: jp.ndarray,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """Computes penalty terms based on distances.

    Args:
        pos_distance: Distance in position space.
        joint_distance: Distance in joint space.
        quat_distance: Quaternion distance.
        too_far_dist: Threshold for position distance penalty.
        bad_pose_dist: Threshold for joint distance penalty.
        bad_quat_dist: Threshold for quaternion distance penalty.
        penalty_pos_distance_scale: Scaling factor for positional penalties as an array.

    Returns:
        Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
            Penalties for being too far, bad pose, bad quaternion,
            and the summed position distance.
    """
    summed_pos_distance = jp.sum((pos_distance * penalty_pos_distance_scale) ** 2)
    too_far = jp.where(summed_pos_distance > too_far_dist, 1.0, 0.0)
    bad_pose = jp.where(joint_distance > bad_pose_dist, 1.0, 0.0)
    bad_quat = jp.where(quat_distance > bad_quat_dist, 1.0, 0.0)
    return too_far, bad_pose, bad_quat, summed_pos_distance


def compute_action_variance_cost(info: dict, var_weight: float) -> jp.ndarray:
    """
    Computes the action variance cost based on the action buffer.

    Args:
        info (dict): Information dictionary containing the action buffer.
        var_weight (float): Weighting factor for the variance cost.

    Returns:
        float: Computed action variance cost.
    """
    # compute windowed variance penalty
    buffer = info["action_buffer"]
    mean_act = jp.mean(buffer, axis=0)
    var_act = jp.mean((buffer - mean_act) ** 2, axis=0)
    var_cost = var_weight * jp.sum(var_act)
    return var_cost


def compute_jerk_cost(
    info: dict, var_window_size: int, jerk_cost_weight: float
) -> jp.ndarray:
    """
    Computes the jerk cost based on the action buffer.

    Args:
        info (dict): dictionary contain additional info about the state
        var_window_size (int): size of the window for variance calculation
        jerk_cost_weight (float): weighting factor for the jerk cost

    Returns:
        float: Computed jerk cost.
    """
    # compute integrated jerk window penalty using JAX-compatible rotation
    buffer = info["action_buffer"]
    action_size = buffer.shape[-1]
    idx = info["buffer_index"]
    # rotate buffer via doubling and dynamic slice to avoid dynamic slicing errors
    doubled = jp.concatenate([buffer, buffer], axis=0)
    ordered = jax.lax.dynamic_slice(doubled, (idx, 0), (var_window_size, action_size))
    jerks = ordered[2:] - 2 * ordered[1:-1] + ordered[:-2]
    jerk_cost = jerk_cost_weight * jp.sum(jerks**2)
    return jerk_cost


def compute_tracking_rewards(
    data: mujoco.MjData,
    reference_frame: ReferenceClip,
    walker: BaseWalker,
    action: jp.ndarray,
    info: dict[str, jp.ndarray],
    reward_config: RewardConfig,
) -> tuple[Union[jp.ndarray, dict[str, jp.ndarray]], ...]:
    """Computes tracking rewards and penalties for motion imitation.

    Args:
        data: Current state MjData object.
        reference_clip: Reference trajectory objct.
        walker: Base walker object.
        action: Current action.
        info: Dictionary of information for logging
        reward_config: Reward configuration object.

    Returns:
        Tuple[float, Dict[str, float]]: Total reward and detailed info dictionary.
    """

    pos_array = data.qpos[:3]
    reference_frame_pos = reference_frame.position
    pos_reward, pos_distance = compute_pos_reward(
        pos_array,
        reference_frame_pos,
        reward_config.pos_reward_weight,
        reward_config.pos_reward_exp_scale,
    )

    quat_array = data.qpos[3:7]
    reference_frame_quat = reference_frame.quaternion
    quat_reward, quat_distance = compute_quat_reward(
        quat_array,
        reference_frame_quat,
        reward_config.quat_reward_weight,
        reward_config.quat_reward_exp_scale,
    )

    joint_array = data.qpos[7:]
    reference_frame_joint = reference_frame.joints
    joint_reward, joint_distance = compute_joint_reward(
        joint_array,
        reference_frame_joint,
        reward_config.joint_reward_weight,
        reward_config.joint_reward_exp_scale,
    )

    angvel_array = data.qvel[3:6]
    reference_frame_angvel = reference_frame.angular_velocity
    angvel_reward = compute_angvel_reward(
        angvel_array,
        reference_frame_angvel,
        reward_config.angvel_reward_weight,
        reward_config.angvel_reward_exp_scale,
    )

    bodypos_array = walker.get_body_positions(data.xpos)
    reference_frame_bodypos = reference_frame.body_positions[walker.body_idxs]
    bodypos_reward = compute_bodypos_reward(
        bodypos_array,
        reference_frame_bodypos,
        reward_config.bodypos_reward_weight,
        reward_config.bodypos_reward_exp_scale,
    )

    endeff_array = walker.get_end_effector_positions(data.xpos)
    reference_frame_endeff = reference_frame.body_positions[walker.endeff_idxs]
    endeff_reward = compute_endeff_reward(
        endeff_array,
        reference_frame_endeff,
        reward_config.endeff_reward_weight,
        reward_config.endeff_reward_exp_scale,
    )

    ctrl_cost = compute_ctrl_cost(action, reward_config.ctrl_cost_weight)
    ctrl_diff_cost = compute_ctrl_diff_cost(
        action, info["prev_ctrl"], reward_config.ctrl_diff_cost_weight
    )

    energy_cost = compute_energy_cost(
        data.qvel[6:],
        data.qfrc_actuator[6:],
        reward_config.energy_cost_weight,
    )

    torso_z = walker.get_torso_position(data.xpos)[2]
    fall = compute_health_penalty(torso_z, reward_config.healthy_z_range)
    too_far, bad_pose, bad_quat, summed_pos_distance = compute_penalty_terms(
        pos_distance,
        joint_distance,
        quat_distance,
        reward_config.too_far_dist,
        reward_config.bad_pose_dist,
        reward_config.bad_quat_dist,
        reward_config.penalty_pos_distance_scale,
    )

    action_variance_cost = compute_action_variance_cost(info, reward_config.var_coeff)
    jerk_cost = compute_jerk_cost(
        info, reward_config.var_window_size, reward_config.jerk_coeff
    )

    # TODO: return a structured dict
    return (
        pos_reward,
        quat_reward,
        joint_reward,
        angvel_reward,
        bodypos_reward,
        endeff_reward,
        ctrl_cost,
        ctrl_diff_cost,
        energy_cost,
        too_far,
        bad_pose,
        bad_quat,
        fall,
        joint_distance,
        summed_pos_distance,
        quat_distance,
        action_variance_cost,
        jerk_cost,
    )
