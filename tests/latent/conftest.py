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


@pytest.fixture
def pretrain_cfg(tmp_path):
    """Tiny pretrain config that completes in <30s."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "walker_name": "rodent",
        "n_joints": 32,
        "reference_data_path": None,  # tests inject synthetic clips
        "clip_length": 20,
        "keep_clips_idx": None,
        "train_ratio": 0.8,
        "seed": 0,
        "window_len": 4,
        "horizon": 2,
        "encoder_layer_sizes": [16],
        "decoder_layer_sizes": [16],
        "predictor_layer_sizes": [16],
        "latent_dim": 4,
        "beta_kl": 0.01,
        "w_pred": 1.0,
        "beta_kl_anneal_steps": 10,
        "batch_size": 8,
        "num_steps": 50,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "log_every": 100,
        "eval_every": 100,
        "viz_every": 1000,
        "viz_n_dims": 4,
        "ckpt_dir": str(tmp_path / "ckpt"),
        "ckpt_every": 100,
        "wandb_enabled": False,
        "wandb_project": "test",
        "wandb_run_name": None,
        "wandb_tags": [],
        "wandb_group": None,
    })
