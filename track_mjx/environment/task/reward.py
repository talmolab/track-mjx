f"""All reward calculation and reward calculation helper functions"""

from jax import numpy as jp
import numpy as np
from typing import Union
from omegaconf import ListConfig
from track_mjx.environment.walker.base import BaseWalker
from track_mjx.io.load import ReferenceClip
from mujoco import MjData

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


def compute_tracking_rewards(
    data: MjData,
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
    )


@struct.dataclass
class RewardConfigReach:
    """Weights and scales for the reaching reward terms.
    Initialized through env_config.reward_weights in the config.yaml file.
    """

    ctrl_cost_weight: float
    ctrl_diff_cost_weight: float
    energy_cost_weight: float
    joint_reward_weight: float
    endeff_reward_weight: float
    joint_reward_exp_scale: float
    endeff_reward_exp_scale: float
    joint_penalty_threshold: float  # Threshold for joint distance penalty
    joint_penalty_weight: float  # Threshold for joint distance penalty
    emg_cost_weight: float  # Weight for emg cost

def compute_penalty_terms_reach(
    joint_distance: jp.ndarray,
    joint_penalty_threshold: float,
    joint_penalty_weight: float,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Computes penalty terms for reaching tasks based on joint distance.

    Args:
        joint_distance: Distance in joint space.
        joint_penalty_threshold: Threshold for joint distance penalty.
        joint_penalty_weight: Weight for the joint penalty.

    Returns:
        Tuple[jp.ndarray, jp.ndarray]: Penalty for bad joint pose and the joint distance.
    """
    bad_pose = jp.where(joint_distance > joint_penalty_threshold, 1.0, 0.0)
    penalty = joint_penalty_weight * bad_pose
    return penalty, joint_distance

def compute_emg_cost(
    action: jp.ndarray, 
    actuator_indices: jp.ndarray,
    emg_values: jp.ndarray,
    valid_mask: jp.ndarray,
    emg_cost_weight: float
) -> jp.ndarray:
    """Compute EMG cost in a fully JAX-compatible way.
    
    Args:
        action: Control actions (muscle activations) for all actuators
        actuator_indices: Indices of actuators that have EMG data (e.g., [3, 5, 8])
        emg_values: EMG values for those specific actuators at the current frame
        valid_mask: Binary mask indicating valid comparisons (1=valid, 0=invalid)
        emg_cost_weight: Weight for EMG cost
    
    Returns:
        jp.ndarray: Weighted EMG cost
    """
    # Get action values for the muscles with EMG data
    action_values = action[actuator_indices]
    
    # Normalize actions to [0,1] to match EMG normalization
    action_values = jp.clip(action_values, 0, 1)
    
    # Calculate squared differences between actions and EMG values
    squared_diffs = jp.square(action_values - emg_values)
    
    # Apply mask to zero out invalid comparisons
    masked_diffs = squared_diffs * valid_mask
    
    # Compute average cost (avoiding division by zero)
    valid_count = jp.maximum(jp.sum(valid_mask), 1.0)
    mean_cost = jp.sum(masked_diffs) / valid_count
    
    # Apply weight
    return emg_cost_weight * mean_cost

def compute_reaching_rewards(
    data: MjData,
    reference_frame: ReferenceClip,
    walker: BaseWalker,
    action: jp.ndarray,
    info: dict[str, jp.ndarray],
    reward_config: RewardConfigReach,
    actuator_indices: jp.ndarray = None,
    emg_values: jp.ndarray = None,
    valid_mask: jp.ndarray = None,
) -> tuple[
    jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray
]:
    """Computes reaching rewards and penalties for motion imitation.

    Args:
        data: Current state MjData object.
        reference_frame: Reference trajectory object.
        walker: Base walker object.
        action: Current action.
        info: Dictionary of information for logging
        reward_config: Reward configuration object.
        actuator_indices: Indices of actuators that have EMG data
        emg_values: EMG values for those actuators at the current frame
        valid_mask: Binary mask indicating valid comparisons (1=valid, 0=invalid)

    Returns:
        Tuple containing: joint_reward, endeff_reward, ctrl_cost, ctrl_diff_cost,
        energy_cost, joint_distance, joint_penalty, emg_cost
    """

    joint_array = data.qpos
    reference_frame_joint = reference_frame.joints
    joint_reward, joint_distance = compute_joint_reward(
        joint_array,
        reference_frame_joint,
        reward_config.joint_reward_weight,
        reward_config.joint_reward_exp_scale,
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
        data.qvel,
        data.qfrc_actuator,
        reward_config.energy_cost_weight,
    )

    joint_penalty, joint_distance = compute_penalty_terms_reach(
        joint_distance,
        reward_config.joint_penalty_threshold,
        reward_config.joint_penalty_weight,
    )

    # EMG cost calculation with safe defaults
    emg_cost = jp.array(0.0)
    
    # Use default zero arrays if None was provided
    safe_indices = jp.array([0], dtype=jp.int32) if actuator_indices is None else actuator_indices
    safe_values = jp.array([0.0]) if emg_values is None else emg_values
    safe_mask = jp.array([0.0]) if valid_mask is None else valid_mask

    emg_cost = compute_emg_cost(
            action=action,
            actuator_indices= actuator_indices,
            emg_values=safe_values,
            valid_mask=safe_mask,
            emg_cost_weight=reward_config.emg_cost_weight
        )

    return (
        joint_reward,
        endeff_reward,
        ctrl_cost,
        ctrl_diff_cost,
        energy_cost,
        joint_distance,
        joint_penalty,
        emg_cost,  # Add this
    )
