"""The imitation entry loads and splits reference clips through the loader's API.

The entry once passed a ``keep_clips_idx`` keyword that the clip loader no
longer takes, which failed only once the entry ran on a GPU. These tests run
the entry's own clip-loading step on CPU against a small clip file written in
the loader's format, and check the keyword it sends against the real function.
"""

import inspect

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf
from vnl_playground import registry
from vnl_playground.tasks.rodent import consts

from tests.scripts._entries import load_entry
from tests.scripts.test_configs_compose import _compose

ENV_NAME = "RodentImitation"
N_CLIPS = 4
N_FRAMES = 12


@pytest.fixture
def entry(monkeypatch):
    return load_entry("imitation", monkeypatch)


def _write_clips(path, n_clips=N_CLIPS, n_frames=N_FRAMES):
    qpos_names = ["root"] * 7 + list(consts.JOINTS)
    xpos_names = ["world", "walker", *consts.BODIES]
    total = n_clips * n_frames
    rng = np.random.default_rng(0)
    qpos = rng.normal(size=(total, len(qpos_names))).astype(np.float32)
    qvel = rng.normal(size=(total, len(qpos_names) - 1)).astype(np.float32)
    xpos = rng.normal(size=(total, len(xpos_names), 3)).astype(np.float32)
    xquat = np.zeros((total, len(xpos_names), 4), np.float32)
    xquat[..., 0] = 1.0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("qpos", data=qpos)
        h5.create_dataset("qvel", data=qvel)
        h5.create_dataset("xpos", data=xpos)
        h5.create_dataset("xquat", data=xquat)
        h5.create_dataset("names_qpos", data=np.asarray(qpos_names, dtype="S"))
        h5.create_dataset("names_xpos", data=np.asarray(xpos_names, dtype="S"))


def _cfg(path, **env_extra):
    return OmegaConf.create(
        {
            "env_config": {
                "reference_data_path": str(path),
                "clip_length": N_FRAMES,
                **env_extra,
            },
            "train_setup": {"train_subset_ratio": 0.5},
        }
    )


def test_clips_load_and_split(entry, tmp_path):
    path = tmp_path / "clips.h5"
    _write_clips(path)
    reference, train = entry["_load_train_clips"](_cfg(path), ENV_NAME, seed=0)
    assert reference.qpos.shape[:2] == (N_CLIPS, N_FRAMES)
    assert train.qpos.shape[:2] == (N_CLIPS // 2, N_FRAMES)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_split_is_seeded_with_the_integer_run_seed(entry, tmp_path, seed):
    """The split ``scripts/train.py`` makes: ``split(ratio, seed=<run seed>)``."""
    path = tmp_path / "clips.h5"
    _write_clips(path, n_clips=12)
    reference, train = entry["_load_train_clips"](_cfg(path), ENV_NAME, seed=seed)
    expected, _ = reference.split(train_ratio=0.5, seed=seed)
    assert train.qpos.shape[0] == 6
    np.testing.assert_array_equal(train.qpos, expected.qpos)


def test_clip_indices_select_source_clips(entry, tmp_path):
    path = tmp_path / "clips.h5"
    _write_clips(path)
    reference, train = entry["_load_train_clips"](
        _cfg(path, clip_indices=[0, 2, 3]), ENV_NAME, seed=0
    )
    assert reference.qpos.shape[0] == 3
    assert train.qpos.shape[0] == 1


def test_retired_keep_clips_idx_is_refused(entry, tmp_path):
    path = tmp_path / "clips.h5"
    _write_clips(path)
    with pytest.raises(ValueError, match="clip_indices"):
        entry["_load_train_clips"](_cfg(path, keep_clips_idx=[0, 1]), ENV_NAME, seed=0)


def test_entry_keywords_exist_on_the_real_loader():
    parameters = inspect.signature(registry.load_reference_clips).parameters
    for name in ("data_path", "n_frames_per_clip", "clip_indices"):
        assert name in parameters
    assert "keep_clips_idx" not in parameters


@pytest.mark.parametrize("name", ["rodent-imitation", "rodent-imitation-intention"])
def test_composed_configs_use_clip_indices(name, monkeypatch):
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/dummy/priors")
    monkeypatch.setenv("DMPO_CKPT_ROOT", "/dummy/checkpoints")
    env_config = _compose(name).env_config
    assert "keep_clips_idx" not in env_config
    assert "clip_indices" in env_config
