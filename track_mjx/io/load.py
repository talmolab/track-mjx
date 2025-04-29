from typing import Union
import h5py
from jax import numpy as jp
import numpy as np
from flax import struct
import yaml
from omegaconf import DictConfig
from pathlib import Path
import hydra
import logging
from dataclasses import dataclass
from typing import Literal


@struct.dataclass
class ReferenceClip:
    """Immutable dataclass defining the trajectory features used in the tracking task."""

    # qpos
    position: jp.ndarray 
    quaternion: jp.ndarray
    joints: jp.ndarray

    # xpos
    body_positions: jp.ndarray

    # velocity (inferred)
    velocity: jp.ndarray
    angular_velocity: jp.ndarray
    joints_velocity: jp.ndarray

    # xquat
    body_quaternions: jp.ndarray
    
    # clip_idx based on the original clip order. Used to recover the metadata
    # for the original clip.
    clip_idx: jp.ndarray | None = None


def load_configs(config_dir: Union[Path, str], config_name: str) -> DictConfig:
    """Initializes configs with hydra.

    Args:
        config_dir ([Path, str]): Absolute path to config directory.

    Returns:
        DictConfig: stac.yaml config to use in run_stac()
    """
    # Initialize Hydra and set the config path
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose the configuration by specifying the config name
        cfg = hydra.compose(config_name=config_name)
        # TODO: Convert to structured config
        # structured_config = OmegaConf.structured(io.Config)
        # OmegaConf.merge(structured_config, cfg)
        # print("Config loaded and validated.")
        return cfg


def load_data(data_path: str):
    """Loads data from the specified path.
    Try to load the data in the stac-mjx format first,
    then try the reference clip format (fly data).
    """
    # TODO: Use a base path given by the config
    data_path = hydra.utils.to_absolute_path(data_path)
    try:
        return make_multiclip_data(data_path)
    except KeyError:
        logging.info(
            f"Loading from stac-mjx format failed. Trying the ReferenceClip format."
        )
        return load_reference_clip_data(data_path)


def make_singleclip_data(traj_data_path):
    """Opens h5 file and makes ReferenceClip based on qpos, qvel, xpos, and xquat.

    Args:
        traj_data_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with h5py.File(traj_data_path, "r+") as data:
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


def make_multiclip_data(traj_data_path, n_frames_per_clip: int | None = None):
    """Creates ReferenceClip object with multiclip tracking data.
    Features have shape = (clips, frames, dims)
    """

    def reshape_frames(arr, clip_len):
        return jp.array(
            arr[()].reshape(arr.shape[0] // clip_len, clip_len, *arr.shape[1:])
        )

    with h5py.File(traj_data_path, "r") as data:
        # Read the config string as yaml in to dict if needed
        if n_frames_per_clip is None:
            yaml_str = data["config"][()]
            yaml_str = yaml_str.decode("utf-8")
            config = yaml.safe_load(yaml_str)
            n_frames_per_clip = config["stac"]["n_frames_per_clip"]

        # Reshape the data to (clips, frames, dims)
        batch_qpos = reshape_frames(data["qpos"], n_frames_per_clip)
        batch_xpos = reshape_frames(data["xpos"], n_frames_per_clip)
        batch_qvel = reshape_frames(data["qvel"], n_frames_per_clip)
        batch_xquat = reshape_frames(data["xquat"], n_frames_per_clip)
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


def load_reference_clip_data(
    filepath: str, group_name: str = "all_clips"
) -> ReferenceClip:
    """Loads data from an HDF5 file containing Datasets for ReferenceClip.
    Args:
        filepath: Path to the HDF5 file.
        group_name: Name of the group containing the datasets.  Defaults to "all_clips".
    Returns:
        A ReferenceClip object containing the loaded data.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the specified group or expected datasets are not found.
    """
    try:
        with h5py.File(filepath, "r") as f:
            if group_name not in f:
                raise KeyError(f"Group '{group_name}' not found in the HDF5 file.")

            group = f[group_name]

            # Load datasets and convert them to JAX arrays.
            data = {}
            for key in [
                "angular_velocity",
                "body_positions",
                "body_quaternions",
                "joints",
                "joints_velocity",
                "position",
                "quaternion",
                "velocity",
            ]:
                if key not in group:
                    raise KeyError(
                        f"Dataset '{key}' not found in group '{group_name}'."
                    )
                data[key] = jp.array(group[key][()])

            # Create and return the ReferenceClip object
            return ReferenceClip(**data)

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except OSError as e:  # Catch more specific HDF5-related errors
        raise OSError(f"Error reading HDF5 file: {filepath} - {e}")


def sub_sample_training_set(train_idx: np.ndarray, train_ratio: float = 0.1):
    """
    Given the indices of the training clips, this function randomly samples a subset
    of the training clips based on the provided ratio without replacement.

    Args:
        train_idx (np.ndarray): Array of indices for the training clips.
        train_ratio (float, optional): Ratio of the training clips to sample. Defaults to 0.1.
    """
    sample_size = int(len(train_idx) * train_ratio)
    sampled_idx = np.random.choice(train_idx, size=sample_size, replace=False)
    sampled_idx.sort()
    return sampled_idx


def select_clips(
    clips: ReferenceClip,
    indices: np.ndarray,
) -> ReferenceClip:
    """
    Selects clips from the ReferenceClip object based on the provided indices.
    The function returns a new ReferenceClip object containing only the selected clips.
    """
    indices = np.array(indices)
    selected_clips = ReferenceClip(
        position=clips.position[indices],
        quaternion=clips.quaternion[indices],
        joints=clips.joints[indices],
        body_positions=clips.body_positions[indices],
        velocity=clips.velocity[indices],
        angular_velocity=clips.angular_velocity[indices],
        joints_velocity=clips.joints_velocity[indices],
        body_quaternions=clips.body_quaternions[indices],
        clip_idx=jp.array(indices[:, np.newaxis]),
    )
    return selected_clips
