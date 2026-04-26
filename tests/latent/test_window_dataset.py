import numpy as np

from track_mjx.agent.latent_ppo.data.window_dataset import make_windows


def test_make_windows_shape():
    n_clips, n_frames, feat_dim = 2, 20, 8
    m = np.arange(n_clips * n_frames * feat_dim, dtype=np.float32).reshape(
        n_clips, n_frames, feat_dim
    )
    inputs, targets = make_windows(m, w=4, n=2)
    # For each clip we get n_frames - w - n + 1 = 20 - 4 - 2 + 1 = 15 windows
    assert inputs.shape == (n_clips * 15, 4, feat_dim)
    assert targets.shape == (n_clips * 15, 2, feat_dim)


def test_make_windows_alignment():
    """target window starts at the frame after the input window ends."""
    m = np.arange(30, dtype=np.float32).reshape(1, 30, 1)
    inputs, targets = make_windows(m, w=3, n=2)
    # First window: input = [0,1,2], target = [3,4]
    np.testing.assert_array_equal(inputs[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(targets[0, :, 0], [3, 4])
    # Second window: input = [1,2,3], target = [4,5]
    np.testing.assert_array_equal(inputs[1, :, 0], [1, 2, 3])
    np.testing.assert_array_equal(targets[1, :, 0], [4, 5])


def test_make_windows_rejects_short_clips():
    m = np.zeros((1, 4, 2), dtype=np.float32)
    # w=4, n=2 needs >= 6 frames per clip
    with __import__("pytest").raises(ValueError, match="too short"):
        make_windows(m, w=4, n=2)
