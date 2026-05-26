"""Tests for the fly named -> legacy converter."""
import h5py


def test_fixture_has_expected_keys(tiny_fly_named_h5):
    with h5py.File(tiny_fly_named_h5) as f:
        assert "all_clips/position" in f
        assert "all_clips/quaternion" in f
        assert f["all_clips/joints"].shape == (2, 5, 36)


import numpy as np
from scripts.convert_fly_reference_to_legacy import build_qpos_qvel


def test_build_qpos_concatenates_root_and_joints(tiny_fly_named_h5):
    import h5py
    with h5py.File(tiny_fly_named_h5) as f:
        position = f["all_clips/position"][()]
        quaternion = f["all_clips/quaternion"][()]
        joints = f["all_clips/joints"][()]
        velocity = f["all_clips/velocity"][()]
        angular_velocity = f["all_clips/angular_velocity"][()]
        joints_velocity = f["all_clips/joints_velocity"][()]

    qpos, qvel = build_qpos_qvel(
        position, quaternion, joints,
        velocity, angular_velocity, joints_velocity,
    )

    # Shape contracts
    assert qpos.shape == (2, 5, 43)
    assert qvel.shape == (2, 5, 42)

    # Layout contracts
    np.testing.assert_array_equal(qpos[..., :3], position)
    np.testing.assert_array_equal(qpos[..., 3:7], quaternion)
    np.testing.assert_array_equal(qpos[..., 7:], joints)
    np.testing.assert_array_equal(qvel[..., :3], velocity)
    np.testing.assert_array_equal(qvel[..., 3:6], angular_velocity)
    np.testing.assert_array_equal(qvel[..., 6:], joints_velocity)
