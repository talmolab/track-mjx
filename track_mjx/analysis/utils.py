"""Utility functions for saving and loading data in HDF5 format."""

import h5py
import numpy as np
import os
import re
import jax.numpy as jnp


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
    elif isinstance(data, jnp.ndarray):  # Convert JAX array to NumPy
        file.create_dataset(group_path, data=np.array(data))
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


def get_aggregate_data(group_path, keys: list[str], clip_idx: int, path: str):
    """
    Get the aggregate data from the hdf5 file
    """
    with h5py.File(path + f'/clip_{clip_idx}.h5', "r") as h5file:
        data = load_from_h5py(h5file, group_path=group_path)
        for key in keys:
            if type(data) == list and type(data[0]) == dict:
                data = [d[key] for d in data]
            elif type(data) == dict:
                data = data[key]
            else:
                raise ValueError("Data structure not supported")
    return data

def extract_clip_info(snippet_path: str):
    """
    Extracts the behavior label and clip number from a snippet filename.
    """
    filename = os.path.basename(snippet_path) 
    name, _ = os.path.splitext(filename) 

    match = re.match(r"([a-zA-Z]+)_(\d+)", name) 
    if match:
        behavior = match.group(1) 
        clip_number = int(match.group(2)) 
        return behavior, clip_number
    else:
        return None, None

def subsample_data(X, sample_size):
    """randomly subsamples the dataset for faster computation"""
    
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    return X[indices]