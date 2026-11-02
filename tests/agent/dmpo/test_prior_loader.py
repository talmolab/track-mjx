"""Tests for the frozen-prior loader, which delegates to a pinned SCAMPER install.

Two groups:

* Artifact tests load real SCAMPER mlp_prior checkpoints from disk and are
  skipped unless ``DMPO_PRIOR_ROOT`` points at a directory holding them (the
  directory names are the run ids used below).
* Unit tests need no artifacts: they check the delegation, the type conversion
  and the install message when SCAMPER is absent.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from track_mjx.agent.dmpo import obs_utils, prior_utils

ROOT = os.environ.get("DMPO_PRIOR_ROOT")
GOLDEN_PATH = Path(__file__).parent / "golden" / "prior_loader_checksums.json"
PRIOR_IDS = ["260407_183223_056788", "260827_153154_576196"]


def needs_prior_artifacts(fn):
    """Mark a test as needing the on-disk priors; skip when they are absent."""
    fn = pytest.mark.skipif(
        ROOT is None, reason="DMPO_PRIOR_ROOT not set (prior artifacts absent)"
    )(fn)
    return pytest.mark.artifact(fn)


def _path_key(entry) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


def _leaf_checksums(tree, prefix: str) -> dict:
    """sha256 of every leaf's raw bytes, keyed by ``prefix/tree/path``."""
    out = {}
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flat:
        arr = np.ascontiguousarray(np.asarray(leaf))
        name = "/".join([prefix, *(_path_key(p) for p in path)])
        out[name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    return out


def _config_checksum(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()


def _latest_step(path: str) -> int:
    return max(
        int(d.rsplit("_", 1)[1])
        for d in os.listdir(path)
        if d.startswith("PriorNetwork_")
    )


@pytest.fixture
def scamper_missing(monkeypatch):
    """Make ``import scamper...`` fail, even for submodules already imported."""
    for name in [m for m in sys.modules if m == "scamper" or m.startswith("scamper.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "scamper", None)


@needs_prior_artifacts
def test_load_gap_naive_prior_latest_step():
    enc, prior, dec, norm, cfg = prior_utils.load_prior_checkpoint(
        f"{ROOT}/260407_183223_056788"
    )
    assert cfg["network_config"]["intention_size"] > 0
    assert set(prior) >= {"hidden_0", "fc2_mean", "fc2_logvar"}
    assert (
        norm.proprioception.mean.shape[-1]
        == cfg["network_config"]["obs_sizes"]["proprioception"]
    )


@needs_prior_artifacts
def test_prior_fn_returns_latent_mean_and_logvar():
    enc, prior, dec, norm, cfg = prior_utils.load_prior_checkpoint(
        f"{ROOT}/260827_153154_576196", step=24
    )
    fn = prior_utils.make_prior_inference_fn(prior, norm, cfg)
    import jax.numpy as jnp

    proprio = jnp.zeros((2, cfg["network_config"]["obs_sizes"]["proprioception"]))
    mean, logvar = fn(proprio)
    assert (
        mean.shape == (2, cfg["network_config"]["intention_size"])
        and logvar.shape == mean.shape
    )


@needs_prior_artifacts
@pytest.mark.parametrize("prior_id", PRIOR_IDS)
def test_loader_matches_vendored_checksums(prior_id):
    """Every restored leaf is bit-identical to what the vendored loader gave."""
    golden = json.loads(GOLDEN_PATH.read_text())["priors"][prior_id]
    path = f"{ROOT}/{prior_id}"
    assert _latest_step(path) == golden["step"]

    enc, prior, dec, norm, cfg = prior_utils.load_prior_checkpoint(path)

    got = {}
    got.update(_leaf_checksums(enc, "encoder"))
    got.update(_leaf_checksums(prior, "prior"))
    got.update(_leaf_checksums(dec, "decoder"))
    got.update(_leaf_checksums(norm, "normalizer"))
    assert got == golden["leaves"]
    assert _config_checksum(cfg) == golden["config_sha256"]


@needs_prior_artifacts
@pytest.mark.parametrize("prior_id", PRIOR_IDS)
def test_loader_returns_dmpo_normalizer_type(prior_id):
    *_, norm, _cfg = prior_utils.load_prior_checkpoint(f"{ROOT}/{prior_id}")
    assert isinstance(norm, obs_utils.DictRunningStatisticsState)


def test_prior_is_the_scamper_prior():
    prior_networks = pytest.importorskip("scamper.agent.mlp_prior.prior_networks")
    assert prior_utils.Prior is prior_networks.Prior


def test_vendored_loader_helpers_are_gone():
    assert not hasattr(prior_utils, "_create_abstract_prior_policy")
    assert not hasattr(prior_utils, "_init_network_params")
    # Kept local: SCAMPER has no logits-only decoder factory.
    assert callable(prior_utils.make_decoder_logits_fn)


def test_loader_delegates_and_converts_the_normalizer(monkeypatch):
    """The SCAMPER loader's own state class becomes the DMPO one, values intact."""
    checkpoint_utils = pytest.importorskip(
        "scamper.agent.task_transfer.walk_rear.checkpoint_utils"
    )
    observation_utils = pytest.importorskip("scamper.agent.observation_utils")

    enc, prior, dec, cfg = {"e": 1}, {"p": 2}, {"d": 3}, {"network_config": {}}
    scamper_norm = observation_utils.DictRunningStatisticsState(
        imitation_target=object(), proprioception=object()
    )
    calls = []

    def fake_load(checkpoint_path, step=None):
        calls.append((checkpoint_path, step))
        return enc, prior, dec, scamper_norm, cfg

    monkeypatch.setattr(checkpoint_utils, "load_prior_checkpoint", fake_load)

    out = prior_utils.load_prior_checkpoint("some/dir", step=7)

    assert calls == [("some/dir", 7)]
    assert out[0] is enc and out[1] is prior and out[2] is dec and out[4] is cfg
    norm = out[3]
    assert type(norm) is obs_utils.DictRunningStatisticsState
    assert norm.imitation_target is scamper_norm.imitation_target
    assert norm.proprioception is scamper_norm.proprioception


def test_missing_scamper_message(scamper_missing):
    install = re.escape("pip install --no-deps -r requirements-dmpo-prior.txt")
    with pytest.raises(ImportError, match=install):
        prior_utils.load_prior_checkpoint("does/not/matter")
    with pytest.raises(ImportError, match=install):
        _ = prior_utils.Prior
