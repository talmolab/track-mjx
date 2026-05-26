"""Shared test fixtures."""
import h5py
import numpy as np
import pytest


@pytest.fixture
def tiny_fly_named_h5(tmp_path):
    """Build a 2-clip x 5-frame synthetic fly named-format H5 in tmp_path.

    Uses physically valid (unit-norm) quaternions in MuJoCo [w,x,y,z] order.
    Joint and body counts match the real fly model (36 hinges, 68 bodies
    matching the H5 source — the current MJCF has 69 bodies but the H5
    captures the older 68-body layout).
    """
    path = tmp_path / "tiny_fly_named.h5"
    rng = np.random.default_rng(0)

    n_clips, n_frames = 2, 5
    n_joints, n_bodies = 36, 68

    def unit_quats(shape):
        q = rng.standard_normal((*shape, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        # Force scalar-first convention by making w positive
        q[..., 0] = np.abs(q[..., 0])
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return q

    with h5py.File(path, "w") as f:
        g = f.create_group("all_clips")
        g.create_dataset("position", data=rng.standard_normal((n_clips, n_frames, 3)).astype(np.float32))
        g.create_dataset("velocity", data=rng.standard_normal((n_clips, n_frames, 3)).astype(np.float32))
        g.create_dataset("quaternion", data=unit_quats((n_clips, n_frames)))
        g.create_dataset("angular_velocity", data=rng.standard_normal((n_clips, n_frames, 3)).astype(np.float32))
        g.create_dataset("joints", data=(0.1 * rng.standard_normal((n_clips, n_frames, n_joints))).astype(np.float32))
        g.create_dataset("joints_velocity", data=rng.standard_normal((n_clips, n_frames, n_joints)).astype(np.float32))
        g.create_dataset("body_positions", data=rng.standard_normal((n_clips, n_frames, n_bodies, 3)).astype(np.float32))
        g.create_dataset("body_quaternions", data=unit_quats((n_clips, n_frames, n_bodies)))
    return path
