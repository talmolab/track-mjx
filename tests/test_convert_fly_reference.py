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


import mujoco
from scripts.convert_fly_reference_to_legacy import resolve_body_names, DEFAULT_FLY_XML


def test_resolve_body_names_matches_mj_forward_on_real_model():
    """Build a known qpos via mj_forward, then verify resolve_body_names recovers
    each body's MJCF name from its world-frame position+quaternion."""
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_FLY_XML))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    data.qpos[:3] = rng.standard_normal(3) * 0.01
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    data.qpos[3:7] = q
    data.qpos[7:] = rng.standard_normal(model.nq - 7) * 0.05
    mujoco.mj_forward(model, data)

    model_xpos = np.array(data.xpos)
    model_xquat = np.array(data.xquat)

    # Take the first 68 bodies as the "H5 body order" — this is the layout
    # of the actual fly_reference_clip.h5 generated before eye-camera additions.
    n_h5 = 68
    h5_xpos = model_xpos[:n_h5]
    h5_xquat = model_xquat[:n_h5]

    names = resolve_body_names(model, model_xpos, model_xquat, h5_xpos, h5_xquat, atol=1e-5)

    expected = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(n_h5)
    ]
    assert names == expected


def test_resolve_body_names_aborts_on_mismatch():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_FLY_XML))
    data = mujoco.MjData(model)
    # Use the same random qpos as the happy-path test so bodies are non-degenerate
    rng = np.random.default_rng(7)
    data.qpos[:3] = rng.standard_normal(3) * 0.01
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    data.qpos[3:7] = q
    data.qpos[7:] = rng.standard_normal(model.nq - 7) * 0.05
    mujoco.mj_forward(model, data)
    model_xpos = np.array(data.xpos)
    model_xquat = np.array(data.xquat)

    # Corrupt H5 frame: shift body 3 (head — uniquely positioned) by 1.0 in x.
    # With the random qpos all bodies except the coincident walker/thorax pair
    # have distinct world positions, so the shifted body will have no match.
    n_h5 = 5
    h5_xpos = model_xpos[:n_h5].copy()
    h5_xquat = model_xquat[:n_h5].copy()
    h5_xpos[3, 0] += 1.0

    import pytest
    with pytest.raises(ValueError, match="no MJCF match"):
        resolve_body_names(model, model_xpos, model_xquat, h5_xpos, h5_xquat, atol=1e-5)


import yaml


def test_convert_end_to_end_produces_valid_legacy_h5(tiny_fly_named_h5, tmp_path):
    """Run the converter on the synthetic fixture and check the output H5
    satisfies the legacy schema."""
    from scripts.convert_fly_reference_to_legacy import convert

    output = tmp_path / "tiny_fly_legacy.h5"
    # Use the real fly MJCF; the fixture is synthetic but shapes match what
    # the converter expects (68 bodies in H5 vs 69 in MJCF — body resolution
    # will tolerate this because the synthetic H5's body data was generated
    # with the same random seed and we relax atol).
    # NOTE: because the synthetic H5's body positions are RANDOM (not derived
    # from mj_forward on a known qpos), this end-to-end test must skip the
    # body-name resolution step. Use convert(..., skip_body_resolution=True).
    convert(
        input_path=str(tiny_fly_named_h5),
        output_path=str(output),
        fly_xml=str(DEFAULT_FLY_XML),
        skip_body_resolution=True,
    )

    import h5py
    with h5py.File(output) as f:
        assert "qpos" in f
        assert "qvel" in f
        assert "xpos" in f
        assert "xquat" in f
        assert "names_qpos" in f
        assert "names_xpos" in f
        assert "config" in f

        # Flat layout: 2 clips * 5 frames = 10
        assert f["qpos"].shape == (10, 43)
        assert f["qvel"].shape == (10, 42)
        assert f["xpos"].shape == (10, 68, 3)
        assert f["xquat"].shape == (10, 68, 4)
        assert f["names_qpos"].shape == (43,)
        # names_xpos size matches H5 body count when skipped: synthesized fallback
        assert f["names_xpos"].shape == (68,)

    # Verify YAML config has snips_order
    with h5py.File(output) as f:
        config = yaml.safe_load(f["config"][()])
    assert "model" in config
    assert config["model"]["SCALE_FACTOR"] == 1.0
    assert len(config["model"]["snips_order"]) == 2
    assert config["model"]["snips_order"][0] == "clip_0000"
