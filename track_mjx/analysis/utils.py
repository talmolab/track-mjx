"""Utility functions for saving and loading nested data structures in HDF5 format.

This module provides functions to serialize and deserialize JAX/NumPy pytrees
(nested dicts, lists, arrays) to HDF5 files, preserving the hierarchical structure.
"""

from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np


def save_to_h5py(file_path: str | Path, data: Any, group_path: str = "/") -> None:
    """Save a pytree (nested dict/list/array structure) to an HDF5 file.

    Recursively traverses the data structure and saves each leaf node as an
    HDF5 dataset. Supports dicts, lists, tuples, NumPy arrays, JAX arrays,
    PyTorch tensors, and objects with __dict__.

    Args:
        file_path: Path to the HDF5 file to create (overwrites if exists).
        data: The data to save. Can be a nested structure of dicts, lists,
            arrays, or objects with public attributes.
        group_path: Root HDF5 group path for saving. Defaults to "/".

    Example:
        >>> data = {"obs": np.zeros((100, 64)), "info": {"episode": 1}}
        >>> save_to_h5py("rollout.h5", data)
    """
    with h5py.File(file_path, "w") as file:
        _recursive_save(file, data, group_path)


def _recursive_save(file: h5py.File, data: Any, group_path: str = "/") -> None:
    """Recursively save data to an HDF5 file.

    Args:
        file: Open HDF5 file handle in write mode.
        data: Data to save at the current group path.
        group_path: Current HDF5 group/dataset path.

    Raises:
        TypeError: If the data type is not supported for HDF5 serialization.
    """
    if data is None:
        return

    if isinstance(data, tuple):
        data = list(data)

    if isinstance(data, dict):
        for key, value in data.items():
            _recursive_save(file, value, f"{group_path}/{key}")

    elif isinstance(data, list):
        for i, item in enumerate(data):
            _recursive_save(file, item, f"{group_path}/{i}")

    elif isinstance(data, (int, float, str, bool, np.ndarray)):
        file.create_dataset(group_path, data=data)

    elif isinstance(data, (np.bytes_, bytes)):
        file.create_dataset(group_path, data=data)

    elif hasattr(data, "numpy"):
        # PyTorch tensors or similar with .numpy() method
        file.create_dataset(group_path, data=data.numpy())

    elif isinstance(data, jax.Array) or hasattr(data, "device_buffer"):
        # JAX arrays (including legacy DeviceArray)
        file.create_dataset(group_path, data=np.array(data))

    elif hasattr(data, "__dict__"):
        # Objects with attributes: save public attributes as a dict
        attr_dict = {k: v for k, v in vars(data).items() if not k.startswith("_")}
        _recursive_save(file, attr_dict, group_path)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def load_from_h5py(file_path: str | Path, group_path: str = "/") -> Any:
    """Load a pytree structure from an HDF5 file.

    Reconstructs the nested dict/list structure that was saved with save_to_h5py.
    Groups with numeric keys are reconstructed as lists; others as dicts.

    Args:
        file_path: Path to the HDF5 file to read.
        group_path: HDF5 group path to load from. Defaults to "/" (entire file).

    Returns:
        The reconstructed data structure (nested dicts, lists, and arrays).

    Example:
        >>> data = load_from_h5py("rollout.h5")
        >>> print(data["obs"].shape)
        (100, 64)
    """
    with h5py.File(file_path, "r") as file:
        return _recursive_load(file, group_path)


def _recursive_load(file: h5py.File, group_path: str = "/") -> Any:
    """Recursively load data from an HDF5 file.

    Args:
        file: Open HDF5 file handle in read mode.
        group_path: Current HDF5 group/dataset path to read from.

    Returns:
        Reconstructed data: arrays for datasets, dicts/lists for groups.

    Raises:
        TypeError: If an unexpected HDF5 object type is encountered.
    """
    group = file[group_path]

    if isinstance(group, h5py.Dataset):
        return group[()]

    elif isinstance(group, h5py.Group):
        keys = list(group.keys())
        if keys and all(k.isdigit() for k in keys):
            # Numeric keys indicate this was originally a list
            return [
                _recursive_load(file, f"{group_path}/{k}")
                for k in sorted(keys, key=int)
            ]
        else:
            # String keys indicate a dict
            return {k: _recursive_load(file, f"{group_path}/{k}") for k in keys}

    else:
        raise TypeError(f"Unsupported HDF5 object type: {type(group)}")
