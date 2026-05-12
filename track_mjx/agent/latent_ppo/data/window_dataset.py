"""Slice (n_clips, n_frames, feat_dim) into (input_window, target_window) pairs."""
import numpy as np


def make_windows(
    motion: np.ndarray, w: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized sliding window builder.

    Parameters
    ----------
    motion : (n_clips, n_frames, feat_dim) float32
        Per-clip frame features.
    w : int
        Encoder input window length.
    n : int
        Predictor output horizon length.

    Returns
    -------
    inputs : (n_clips * (n_frames - w - n + 1), w, feat_dim)
    targets : (n_clips * (n_frames - w - n + 1), n, feat_dim)
    """
    if motion.ndim != 3:
        raise ValueError(f"expected (n_clips, n_frames, feat_dim), got {motion.shape}")
    n_clips, n_frames, feat_dim = motion.shape
    n_windows_per_clip = n_frames - w - n + 1
    if n_windows_per_clip <= 0:
        raise ValueError(
            f"clips too short: n_frames={n_frames}, w+n={w + n}"
        )
    starts = np.arange(n_windows_per_clip)
    in_idx = starts[:, None] + np.arange(w)            # (W, w)
    tgt_idx = starts[:, None] + w + np.arange(n)       # (W, n)
    # Broadcast across clips
    inputs = motion[:, in_idx, :].reshape(-1, w, feat_dim)
    targets = motion[:, tgt_idx, :].reshape(-1, n, feat_dim)
    return inputs.astype(np.float32), targets.astype(np.float32)
