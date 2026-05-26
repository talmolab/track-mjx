"""Tests for the fly named -> legacy converter."""
import h5py


def test_fixture_has_expected_keys(tiny_fly_named_h5):
    with h5py.File(tiny_fly_named_h5) as f:
        assert "all_clips/position" in f
        assert "all_clips/quaternion" in f
        assert f["all_clips/joints"].shape == (2, 5, 36)
