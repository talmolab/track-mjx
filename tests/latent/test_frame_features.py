import numpy as np

from track_mjx.agent.latent_ppo.data.frame_features import (
    MOTION_FRAME_DIM,
    extract_motion_frames,
)


def test_motion_frame_dim_for_rat():
    # 3 (p) + 4 (theta) + 6 (v) + 32 (q) + 32 (qdot) = 77
    assert MOTION_FRAME_DIM(n_joints=32) == 77


def test_extract_shapes(synthetic_clips):
    m = extract_motion_frames(synthetic_clips, n_joints=32)
    assert m.shape == (synthetic_clips.n_clips,
                      synthetic_clips.n_frames,
                      77)
    assert m.dtype == np.float32


def test_extract_concatenation_order(synthetic_clips):
    m = extract_motion_frames(synthetic_clips, n_joints=32)
    c = synthetic_clips
    # p = qpos[..., :3]
    np.testing.assert_array_equal(m[..., :3], c.qpos[..., :3])
    # theta = qpos[..., 3:7]
    np.testing.assert_array_equal(m[..., 3:7], c.qpos[..., 3:7])
    # v = qvel[..., :6]
    np.testing.assert_array_equal(m[..., 7:13], c.qvel[..., :6])
    # q = qpos[..., 7:]
    np.testing.assert_array_equal(m[..., 13:45], c.qpos[..., 7:])
    # qdot = qvel[..., 6:]
    np.testing.assert_array_equal(m[..., 45:77], c.qvel[..., 6:])
