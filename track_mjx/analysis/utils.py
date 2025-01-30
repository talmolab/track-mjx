"""Utility functions for saving and loading data in HDF5 format."""

import h5py
import numpy as np


def save_to_h5py(file, data, group_path="/"):
    """
    Save a pytree (like a dictionary) into an HDF5 file.

    Args:
        file (h5py.File): An open HDF5 file object.
        data: The data to save (can be a dictionary, list, etc.).
        group_path (str): The HDF5 group path for saving the data.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            sub_group_path = f"{group_path}/{key}"
            save_to_h5py(file, value, sub_group_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            sub_group_path = f"{group_path}/{i}"
            save_to_h5py(file, item, sub_group_path)
    elif isinstance(data, (int, float, str, bool, np.ndarray)):
        file.create_dataset(group_path, data=data)
    elif hasattr(data, "numpy"):  # For NumPy arrays or PyTorch tensors
        file.create_dataset(group_path, data=data.numpy())
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def load_from_h5py(file, group_path="/"):
    """
    Load a pytree structure from an HDF5 file.

    Args:
        file (h5py.File): An open HDF5 file object.
        group_path (str): The HDF5 group path to read data from.

    Returns:
        The reconstructed data structure.
    """
    group = file[group_path]
    if isinstance(group, h5py.Dataset):
        return group[()]  # Read dataset value
    elif isinstance(group, h5py.Group):
        if all(k.isdigit() for k in group.keys()):  # Likely a list
            return [load_from_h5py(file, f"{group_path}/{k}") for k in sorted(group.keys(), key=int)]
        else:  # Dictionary-like group
            return {k: load_from_h5py(file, f"{group_path}/{k}") for k in group.keys()}
    else:
        raise TypeError(f"Unsupported group type: {type(group)}")


# # Example usage
# with h5py.File("clip1_rollout.h5", "r") as h5file:
#     loaded_data = load_from_h5py(h5file)
