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
from typing import Tuple
import re


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
    original_clip_idx: jp.ndarray | None = None


@struct.dataclass
class ReferenceClipReach:
    """Immutable dataclass defining the trajectory features used in the reaching task.
    This is a simplified version focused on reaching targets rather than full body tracking.
    """

    # qpos - joint positions (no root body for reaching tasks)
    joints: jp.ndarray

    # xpos - body positions (for end effector tracking)
    body_positions: jp.ndarray

    # velocity - joint velocities (no root body for reaching tasks)
    joints_velocity: jp.ndarray

    # clip_idx based on the original clip order. Used to recover the metadata
    # for the original clip.
    original_clip_idx: jp.ndarray | None = None


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


def load_reaching_data(data_path: str):
    """Loads reaching data from the specified path.
    If data_path contains multiple comma-separated file paths, loads them as separate clips.
    """
    # TODO: Use a base path given by the config
    data_path = hydra.utils.to_absolute_path(data_path)
    
    # Check if data_path contains multiple files (comma-separated)
    if ',' in data_path:
        file_paths = [path.strip() for path in data_path.split(',')]
        return load_multiple_reaching_files(file_paths)
    else:
        # Try multiclip format first, then single clip format
        try:
            return make_multiclip_reaching_data(data_path)
        except (KeyError, ValueError):
            # Fall back to single clip format
            return make_reaching_data(data_path)


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


def make_reaching_data(traj_data_path):
    """Opens h5 file and makes ReferenceClipReach based on qpos, qvel, xpos.

    Args:
        traj_data_path: Path to the H5 file containing reaching data

    Returns:
        ReferenceClipReach: Reaching trajectory data
    """
    with h5py.File(traj_data_path, "r") as data:
        qpos = jp.array(data["qpos"][()])
        qvel = jp.array(data["qvel"][()])
        xpos = jp.array(data["xpos"][()])
        
        # For reaching tasks, qpos contains only joint positions (no root)
        joints = qpos  # All of qpos is joint positions
        
        # For reaching tasks, qvel contains only joint velocities (no root)
        joints_velocity = qvel  # All of qvel is joint velocities
        
        # Get body positions (for end effector tracking)
        body_positions = xpos
        
        return ReferenceClipReach(
            joints=joints,
            body_positions=body_positions,
            joints_velocity=joints_velocity,
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

def load_multiple_reaching_files(file_paths: list[str]) -> ReferenceClipReach:
    """Loads reaching data from multiple H5 files and combines them as separate clips.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        ReferenceClipReach: Combined data from all files as separate clips
    """
    all_clips = []
    
    for i, file_path in enumerate(file_paths):
        file_path = file_path.strip()
        try:
            # Try multiclip format first, then single clip format
            try:
                clip_data = make_multiclip_reaching_data(file_path)
            except (KeyError, ValueError):
                # Fall back to single clip format
                clip_data = make_reaching_data(file_path)
            all_clips.append(clip_data)
        except Exception as e:
            logging.warning(f"Failed to load reaching file {file_path}: {e}")
            continue
    
    if not all_clips:
        raise ValueError("No valid reaching files could be loaded from the provided paths")
    
    # Combine all clips into a single ReferenceClipReach
    return combine_reaching_clips(all_clips)

def make_multiclip_reaching_data(traj_data_path, n_frames_per_clip: int | None = None):
    """Creates ReferenceClipReach object with multiclip reaching data.
    Features have shape = (clips, frames, dims)
    """

    def reshape_frames(arr, clip_len):
        return jp.array(
            arr[()].reshape(arr.shape[0] // clip_len, clip_len, *arr.shape[1:])
        )

    with h5py.File(traj_data_path, "r") as data:
        # Read the config string as yaml in to dict if needed
        if n_frames_per_clip is None:
            try:
                yaml_str = data["config"][()]
                yaml_str = yaml_str.decode("utf-8")
                config = yaml.safe_load(yaml_str)
                n_frames_per_clip = config["stac"]["n_frames_per_clip"]
            except (KeyError, ValueError):
                # If no config, assume 100 frames per clip (common for reaching data)
                n_frames_per_clip = 100

        # Reshape the data to (clips, frames, dims)
        batch_qpos = reshape_frames(data["qpos"], n_frames_per_clip)
        batch_xpos = reshape_frames(data["xpos"], n_frames_per_clip)
        batch_qvel = reshape_frames(data["qvel"], n_frames_per_clip)
        
        return ReferenceClipReach(
            joints=batch_qpos,  # All of qpos is joint positions for reaching
            body_positions=batch_xpos,
            joints_velocity=batch_qvel,  # All of qvel is joint velocities for reaching
        )

def combine_reaching_clips(clips: list[ReferenceClipReach]) -> ReferenceClipReach:
    """Combines multiple ReferenceClipReach objects into a single ReferenceClipReach.
    
    Args:
        clips: List of ReferenceClipReach objects to combine
        
    Returns:
        ReferenceClipReach: Combined clips with shape (total_clips, frames, dims)
    """
    if not clips:
        raise ValueError("No reaching clips provided to combine")
    
    # Concatenate all clips along the first dimension (clip dimension)
    combined_joints = jp.concatenate([clip.joints for clip in clips], axis=0)
    combined_body_positions = jp.concatenate([clip.body_positions for clip in clips], axis=0)
    combined_joints_velocity = jp.concatenate([clip.joints_velocity for clip in clips], axis=0)
    
    # Create original_clip_idx to track which file each clip came from
    clip_indices = []
    current_idx = 0
    for i, clip in enumerate(clips):
        num_clips_in_file = clip.joints.shape[0]
        clip_indices.extend([i] * num_clips_in_file)
        current_idx += num_clips_in_file
    
    original_clip_idx = jp.array(clip_indices)[:, jp.newaxis]
    
    return ReferenceClipReach(
        joints=combined_joints,
        body_positions=combined_body_positions,
        joints_velocity=combined_joints_velocity,
        original_clip_idx=original_clip_idx,
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


def generate_train_test_split(
    data: ReferenceClip | ReferenceClipReach,
    test_ratio: float = 0.1,
) -> Tuple[ReferenceClip | ReferenceClipReach, ReferenceClip | ReferenceClipReach]:
    """
    Generates a train-test split of the clips based on the provided ratio.
    The split is done by randomly sampling clips from the metadata list.
    The function returns two objects: one for training and one for testing.

    Args:
        data (ReferenceClip | ReferenceClipReach): The object containing the clips to be split.
        test_ratio (float, optional): ratio of the test set. Defaults to 0.1.

    Returns:
        Tuple: training set and testing set as ReferenceClip or ReferenceClipReach objects.
    """
    if isinstance(data, ReferenceClipReach):
        num_clips = data.joints.shape[0]
    else:
        num_clips = data.position.shape[0]
    
    indices = np.arange(num_clips)
    test_idx = np.random.choice(
        indices, size=int(num_clips * test_ratio), replace=False
    )
    train_idx = indices[~np.isin(indices, test_idx)]
    train_idx.sort()
    test_idx.sort()
    train_set = select_clips(data, train_idx)
    test_set = select_clips(data, test_idx)
    return train_set, test_set


def select_clips(
    clips: ReferenceClip | ReferenceClipReach,
    indices: np.ndarray,
) -> ReferenceClip | ReferenceClipReach:
    """
    Selects clips from the ReferenceClip or ReferenceClipReach object based on the provided indices.
    The function returns a new object containing only the selected clips.
    """
    indices = np.array(indices)
    
    if isinstance(clips, ReferenceClipReach):
        selected_clips = ReferenceClipReach(
            joints=clips.joints[indices],
            body_positions=clips.body_positions[indices],
            joints_velocity=clips.joints_velocity[indices],
            original_clip_idx=jp.array(indices[:, jp.newaxis]),
        )
    else:
        selected_clips = ReferenceClip(
            position=clips.position[indices],
            quaternion=clips.quaternion[indices],
            joints=clips.joints[indices],
            body_positions=clips.body_positions[indices],
            velocity=clips.velocity[indices],
            angular_velocity=clips.angular_velocity[indices],
            joints_velocity=clips.joints_velocity[indices],
            body_quaternions=clips.body_quaternions[indices],
            original_clip_idx=jp.array(indices[:, jp.newaxis]),
        )
    return selected_clips


def load_clips_metadata(traj_data_path: str) -> list[Tuple[str, int]]:
    """
    Loads the metadata of the clips from the specified trajectory data path.
    the metadata includes the behavior groups and number of each clip.
    This methods is specific to the stac-mjx format of the rodent data.

    Args:
        traj_data_path (str): Path to the trajectory data file.

    Returns:
        list[Tuple[str, int]]: List of tuples containing the clip name and number.
    """
    with h5py.File(traj_data_path, "r") as data:
        # Read the config string as yaml in to dict
        yaml_str = data["config"][()]
        yaml_str = yaml_str.decode("utf-8")
        config = yaml.safe_load(yaml_str)
    pattern = re.compile(r"/([^/]+)_([0-9]+)\.p$")
    clip_metadata = []
    for path in config["model"]["snips_order"]:
        match = pattern.search(path)
        if match:
            name, number = match.groups()
            clip_metadata.append((name, int(number)))
    return clip_metadata


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