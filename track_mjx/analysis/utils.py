"""Utility functions for saving and loading data in HDF5 format."""

import h5py
import jax
import numpy as np
from pathlib import Path


def save_to_h5py(file_path: str | Path, data, group_path="/"):
    """
    Save a pytree (like a dictionary) into an HDF5 file.

    Args:
        file (h5py.File): An open HDF5 file object.
        data: The data to save (can be a dictionary, list, etc.).
        group_path (str): The HDF5 group path for saving the data.
    """

    with h5py.File(file_path, "w") as file:
        recursive_dict_to_h5py(file, data, group_path)


def recursive_dict_to_h5py(file, data, group_path="/"):
    if isinstance(data, dict):
        for key, value in data.items():
            sub_group_path = f"{group_path}/{key}"
            recursive_dict_to_h5py(file, value, sub_group_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            sub_group_path = f"{group_path}/{i}"
            recursive_dict_to_h5py(file, item, sub_group_path)
    elif isinstance(data, (int, float, str, bool, np.ndarray)):
        file.create_dataset(group_path, data=data)
    elif hasattr(data, "numpy"):  # For NumPy arrays or PyTorch tensors
        file.create_dataset(group_path, data=data.numpy())
    elif isinstance(data, jax.Array) or hasattr(
        data, "device_buffer"
    ):  # For JAX DeviceArrays
        file.create_dataset(group_path, data=np.array(data))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def load_from_h5py(file_path: str | Path, group_path="/"):
    """
    Load a pytree structure from an HDF5 file.

    Args:
        file (h5py.File): An open HDF5 file object.
        group_path (str): The HDF5 group path to read data from.

    Returns:
        The reconstructed data structure.
    """
    with h5py.File(file_path, "r") as file:
        return recursive_load_from_h5py(file, group_path)


def recursive_load_from_h5py(file, group_path="/"):
    group = file[group_path]
    if isinstance(group, h5py.Dataset):
        return group[()]  # Read dataset value
    elif isinstance(group, h5py.Group):
        if all(k.isdigit() for k in group.keys()):  # Likely a list
            return [
                recursive_load_from_h5py(file, f"{group_path}/{k}")
                for k in sorted(group.keys(), key=int)
            ]
        else:  # Dictionary-like group
            return {
                k: recursive_load_from_h5py(file, f"{group_path}/{k}")
                for k in group.keys()
            }
    else:
        raise TypeError(f"Unsupported group type: {type(group)}")
