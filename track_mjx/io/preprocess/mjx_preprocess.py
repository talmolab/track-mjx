"""Preprocess mocap data for mjx"""

import jax
from jax import jit
from jax import numpy as jp
from flax import struct

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx
from mujoco.mjx._src import smooth

# import preprocessing.transformations as tr
from track_mjx.io.preprocess import transformations as tr

from collections import defaultdict
from typing import Text, Union, List
import h5py
import pickle


@struct.dataclass
class ReferenceClip:
    """This dataclass is used to store the trajectory in the env."""

    # qpos
    position: jp.ndarray = None
    quaternion: jp.ndarray = None
    joints: jp.ndarray = None

    # xpos
    body_positions: jp.ndarray = None

    # velocity (inferred)
    velocity: jp.ndarray = None
    joints_velocity: jp.ndarray = None
    angular_velocity: jp.ndarray = None

    # xquat
    body_quaternions: jp.ndarray = None


def process_clip_to_train(
    stac_path: Text,
    mjcf_path: str = "./assets/rodent.xml",
    scale_factor: float = 0.9,
    start_step: int = 0,
    clip_length: int = 250,
    max_qvel: float = 20.0,
    dt: float = 0.02,
):
    """Process clip function for ../train.py.
    Just mujoco and data setup then calls process_clip

    Args:
        stac_path (Text): _description_
        save_file (Text): _description_
        scale_factor (float, optional): _description_. Defaults to 0.9.
        start_step (int, optional): _description_. Defaults to 0.
        clip_length (int, optional): _description_. Defaults to 250.
        max_qvel (float, optional): _description_. Defaults to 20.0.
        dt (float, optional): _description_. Defaults to 0.02.
        ref_steps (Tuple, optional): _description_. Defaults to (1, 2, 3, 4, 5, 6, 7, 8, 9, 10).
    """
    # Load mocap data from a file.
    with open(stac_path, "rb") as file:
        d = pickle.load(file)
        mocap_qpos = jp.array(d["qpos"])[start_step : start_step + clip_length]

    # Load rodent mjcf and rescale, then get the mj_model from that.
    # TODO: make this all work in mjx? james cotton did rescaling with mjx model:
    # https://github.com/peabody124/BodyModels/blob/f6ef1be5c5d4b7e51028adfc51125e510c13bcc2/body_models/biomechanics_mjx/forward_kinematics.py#L92
    # TODO: Set this up outside of this function as it only needs to be done once anyway
    root = mjcf.from_path(mjcf_path)

    # rescale a rodent model.
    rescale.rescale_subtree(
        root,
        scale_factor,
        scale_factor,
    )
    mj_model = mjcf.Physics.from_mjcf_model(root).model.ptr
    mj_data = mujoco.MjData(mj_model)

    # Initialize MuJoCo model and data structures & place into GPU
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    return process_clip(mocap_qpos, mjx_model, mjx_data, max_qvel=max_qvel, dt=dt)


def process_clip(
    mocap_qpos,
    mjx_model,
    mjx_data,
    max_qvel: float = 20.0,
    dt: float = 0.02,
):
    """Process a set of joint angles into the features that
       the referenced trajectory is composed of. This function
       will process only one clip. Rodent only for now.

    Args:
        stac_path (Text): _description_
        save_file (Text): _description_
        scale_factor (float, optional): _description_. Defaults to 0.9.
        start_step (int, optional): _description_. Defaults to 0.
        clip_length (int, optional): _description_. Defaults to 250.
        max_qvel (float, optional): _description_. Defaults to 20.0.
        dt (float, optional): _description_. Defaults to 0.02.
        ref_steps (Tuple, optional): _description_. Defaults to (1, 2, 3, 4, 5, 6, 7, 8, 9, 10).
    """

    # Feature logic for a single clip here
    clip = ReferenceClip()

    clip = extract_features(mjx_model, mjx_data, clip, mocap_qpos)
    # Padding for velocity corner case.
    mocap_qpos = jp.concatenate([mocap_qpos, mocap_qpos[-1, jp.newaxis, :]], axis=0)

    # Calculate qvel, clip.
    mocap_qvel = compute_velocity_from_kinematics(mocap_qpos, dt)
    vels = mocap_qvel[:, 6:]
    clipped_vels = jp.clip(vels, -max_qvel, max_qvel)

    mocap_qvel = mocap_qvel.at[:, 6:].set(clipped_vels)
    clip = clip.replace(
        velocity=mocap_qvel[:, :3],
        angular_velocity=mocap_qvel[:, 3:6],
        joints_velocity=mocap_qvel[:, 6:],
    )

    return clip


@jit
def extract_features(mjx_model, mjx_data, clip, mocap_qpos):
    def f(mjx_data, qpos):
        mjx_data = set_position(mjx_model, mjx_data, qpos)
        qpos = mjx_data.qpos
        xpos = mjx_data.xpos
        xquat = mjx_data.xquat
        return mjx_data, (qpos[:3], qpos[3:7], qpos[7:], xpos, xquat)

    mjx_data, (position, quaternion, joints, body_positions, body_quaternions) = (
        jax.lax.scan(
            f,
            mjx_data,
            mocap_qpos,
        )
    )

    # Add features to ReferenceClip
    return clip.replace(
        position=position,
        quaternion=quaternion,
        joints=joints,
        body_positions=body_positions,
        body_quaternions=body_quaternions,
    )


def kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """jit compiled forward kinematics

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.kinematics(mjx_model, mjx_data)


@jit
def set_position(
    mjx_model: mjx.Model, mjx_data: mjx.Data, qpos: jp.ndarray
) -> mjx.Data:
    """Sets the qpos and performs forward kinematics (zeros for qvel)

    Args:
        mjx_model (mjx.Model): _description_
        mjx_data (mjx.Data): _description_
        qpos (jp.Array): _description_

    Returns:
        mjx.Data: _description_
    """
    qvel = jp.zeros((mjx_model.nv,))
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = kinematics(mjx_model, mjx_data)
    return mjx_data


@jit
def compute_velocity_from_kinematics(
    qpos_trajectory: jp.ndarray, dt: float
) -> jp.ndarray:
    """Computes velocity trajectory from position trajectory.

    Args:
        qpos_trajectory (jp.ndarray): trajectory of qpos values T x ?
          Note assumes has freejoint as the first 7 dimensions
        dt (float): timestep between qpos entries

    Returns:
        jp.ndarray: Trajectory of velocities.
    """
    qvel_translation = (qpos_trajectory[1:, :3] - qpos_trajectory[:-1, :3]) / dt
    qvel_gyro = []
    for t in range(qpos_trajectory.shape[0] - 1):
        normed_diff = tr.quat_diff(qpos_trajectory[t, 3:7], qpos_trajectory[t + 1, 3:7])
        normed_diff /= jp.linalg.norm(normed_diff)
        angle = tr.quat_to_axisangle(normed_diff)
        qvel_gyro.append(angle / dt)
    qvel_gyro = jp.stack(qvel_gyro)
    qvel_joints = (qpos_trajectory[1:, 7:] - qpos_trajectory[:-1, 7:]) / dt
    return jp.concatenate([qvel_translation, qvel_gyro, qvel_joints], axis=1)


def save_reference_clip_to_h5(
    filename: str, clip_names: Union[List[str], str], reference_clip: ReferenceClip
):
    """Save the contents of a ReferenceClip object to an .h5 file.
    Handles single clip and multiple clips

    Args:
        filename (str):  The name of the .h5 file to save to.
        clip_names (Union[List[str], str]): If multiclip, a list of clip names.
            If single clip, one name as a string
        reference_clip (ReferenceClip): The ReferenceClip object to save.
    """
    assert isinstance(clip_names, str) or isinstance(clip_names, list)

    with h5py.File(filename, "w") as hf:
        if isinstance(clip_names, str):
            for attr, value in reference_clip.__dict__.items():
                # Create a group for the clip and add the attributes as Datasets
                group_name = f"{clip_names}/{attr}"
                hf.create_dataset(group_name, data=value)
        else:
            for i, clip_name in enumerate(clip_names):
                for attr, value in reference_clip.__dict__.items():
                    # Create a group for each clip and add the attributes as Datasets
                    group_name = f"{clip_name}/{attr}"
                    hf.create_dataset(group_name, data=value[i])


def load_reference_clip_from_h5(filename: str, clip_names: Union[List[str], str]):
    """
    Load the contents of an .h5 file into a ReferenceClip object.

    Args:
        filename (str): The name of the .h5 file to load from.
        clip_names (Union[List[str], str]): The order of clips given by a list of names

    Returns:
        ReferenceClip: The reconstructed ReferenceClip object.
    """
    assert isinstance(clip_names, str) or isinstance(clip_names, list)

    if isinstance(clip_names, str):
        clip_names = [clip_names]

    aggregated = defaultdict(lambda: [])
    with h5py.File(filename, "r") as hf:
        clip = ReferenceClip()
        # Get lists of arrays for each feature, using the given clip order
        for clip_name in clip_names:
            for attr in clip.__dict__.keys():
                if f"{clip_name}/{attr}" in hf:
                    aggregated[attr].append(hf[f"{clip_name}/{attr}"][:])

        # Stack them as jax arrays
        for key in aggregated.keys():
            aggregated[key] = jp.stack(aggregated[key])

        # Set the values in the ReferenceClip
        clip = clip.replace(**aggregated)

        return clip
