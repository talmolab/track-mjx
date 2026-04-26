"""Shared fixtures for latent_ppo tests."""
import numpy as np
import pytest


@pytest.fixture
def synthetic_clips():
    """Tiny synthetic 'clips' object with the legacy flat layout used by ReferenceClips.

    n_clips=2, n_frames=20, 32 joints (matches rat).
    Returns a SimpleNamespace with .qpos, .qvel, .xpos, .xquat fields.
    """
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    n_clips, n_frames, n_joints = 2, 20, 32
    qpos = rng.standard_normal((n_clips, n_frames, 7 + n_joints)).astype(np.float32)
    qvel = rng.standard_normal((n_clips, n_frames, 6 + n_joints)).astype(np.float32)
    xpos = rng.standard_normal((n_clips, n_frames, 18, 3)).astype(np.float32)
    xquat = rng.standard_normal((n_clips, n_frames, 18, 4)).astype(np.float32)
    return SimpleNamespace(qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat,
                           n_clips=n_clips, n_frames=n_frames, n_joints=n_joints)
