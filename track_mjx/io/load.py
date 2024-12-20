import h5py
import numpy as np
import jax
from jax import numpy as jp
from flax import struct


@struct.dataclass
class ReferenceClip:
    """Immutable dataclass defining the trajectory features used in the tracking task."""

    # qpos
    position: jp.ndarray = None
    quaternion: jp.ndarray = None
    joints: jp.ndarray = None

    # xpos
    body_positions: jp.ndarray = None

    # velocity (inferred)
    velocity: jp.ndarray = None
    angular_velocity: jp.ndarray = None
    joints_velocity: jp.ndarray = None

    # xquat
    body_quaternions: jp.ndarray = None


def make_singleclip_data(traj_data_path):
    """Opens h5 file and makes ReferenceClip based on qpos, qvel, xpos, and xquat.

    Args:
        traj_data_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with h5py.File(traj_data_path, "r") as data:
        qpos = jp.array(data["qpos"][()])
        qvel = jp.array(data["qvel"][()])
        xpos = jp.array(data["xpos"][()])
        xquat = jp.array(data["xquat"][()])
        return (
            ReferenceClip(
                position=qpos[:, :3],
                quaternion=qpos[:, 3:7],
                joints=qpos[:, 7:],
                body_positions=xpos,
                velocity=qvel[:, :3],
                angular_velocity=qvel[:, 3:6],
                joints_velocity=qvel[:, 6:],
                body_quaternions=xquat,
            ),
        )


def make_multiclip_data(traj_data_path):
    """Creates ReferenceClip object with multiclip tracking data.
    Features have shape = (clips, frames, dims)
    """

    def reshape_frames(arr, clip_len):
        return jp.array(
            arr[()].reshape(arr.shape[0] // clip_len, clip_len, *arr.shape[1:])
        )

    with h5py.File(traj_data_path, "r") as data:
        clip_len = data["stac"]["n_frames_per_clip"][()]
        batch_qpos = reshape_frames(data["qpos"], clip_len)
        batch_xpos = reshape_frames(data["xpos"], clip_len)
        batch_qvel = reshape_frames(data["qvel"], clip_len)
        batch_xquat = reshape_frames(data["xquat"], clip_len)
        return ReferenceClip(
            position=batch_qpos[:, :, :3],
            quaternion=batch_qpos[:, :, 3:7],
            joints=batch_qpos[:, :, 7:],
            body_positions=batch_xpos,
            velocity=batch_qvel[:, :, :3],
            angular_velocity=batch_qvel[:, :, 3:6],
            joints_velocity=batch_qvel[:, :, 6:],
            body_quaternions=batch_xquat,
        )
