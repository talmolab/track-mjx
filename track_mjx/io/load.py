from typing import Union
import h5py
from jax import numpy as jp
from flax import struct
import yaml
from omegaconf import DictConfig
from pathlib import Path
import hydra


@struct.dataclass
class ReferenceClip:
    """Immutable dataclass defining the trajectory features used in the tracking task."""

    # qpos
    # position: jp.ndarray = None
    # quaternion: jp.ndarray = None
    joints: jp.ndarray = None

    # xpos
    body_positions: jp.ndarray = None

    # velocity (inferred)
    # velocity: jp.ndarray = None
    # angular_velocity: jp.ndarray = None
    joints_velocity: jp.ndarray = None

    # xquat
    body_quaternions: jp.ndarray = None


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
                # position=qpos[:, :3],
                # quaternion=qpos[:, 3:7],
                joints=qpos,
                body_positions=xpos,
                # velocity=qvel[:, :3],
                # angular_velocity=qvel[:, 3:6],
                joints_velocity=qvel,
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
        # Read the config string as yaml in to dict
        yaml_str = data["config"][()]
        yaml_str = yaml_str.decode("utf-8")
        config = yaml.safe_load(yaml_str)
        clip_len = config["stac"]["n_frames_per_clip"]

        # Reshape the data to (clips, frames, dims)
        batch_qpos = reshape_frames(data["qpos"], clip_len)
        batch_xpos = reshape_frames(data["xpos"], clip_len)
        batch_qvel = reshape_frames(data["qvel"], clip_len)
        batch_xquat = reshape_frames(data["xquat"], clip_len)
        return ReferenceClip(
            # position=batch_qpos[:, :, :3],
            # quaternion=batch_qpos[:, :, 3:7],
            joints=batch_qpos,
            body_positions=batch_xpos,
            # velocity=batch_qvel[:, :, :3],
            # angular_velocity=batch_qvel[:, :, 3:6],
            joints_velocity=batch_qvel,
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
                # "angular_velocity",
                "body_positions",
                "body_quaternions",
                "joints",
                "joints_velocity",
                # "position",
                # "quaternion",
                # "velocity",
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
