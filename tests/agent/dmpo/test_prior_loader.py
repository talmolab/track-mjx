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
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from scamper.agent import observation_utils
from scamper.agent.mlp_prior import prior_networks
from scamper.agent.task_transfer.walk_rear import checkpoint_utils

from track_mjx.agent.dmpo import obs_utils, prior_utils

ROOT = os.environ.get("DMPO_PRIOR_ROOT")
GOLDEN_PATH = Path(__file__).parent / "golden" / "prior_loader_checksums.json"
PRIOR_IDS = ["260407_183223_056788", "260827_153154_576196"]
REQUIREMENTS_PATH = Path(__file__).parents[3] / "requirements-dmpo-prior.txt"
# Tiny config: enough for the factories to build their closures (no params needed).
TINY_CONFIG = {
    "network_config": {
        "action_size": 2,
        "intention_size": 3,
        "decoder_layer_sizes": [4],
        "prior_layer_sizes": [4],
    }
}


def needs_prior_artifacts(fn):
    """Mark a test as needing the on-disk priors; skip when they are absent."""
    fn = pytest.mark.skipif(
        ROOT is None, reason="DMPO_PRIOR_ROOT not set (prior artifacts absent)"
    )(fn)
    return pytest.mark.artifact(fn)


def _prior_path(prior_id: str) -> str:
    """Path of one staged prior; skip the test if that prior is not staged."""
    path = Path(ROOT) / prior_id
    if not path.is_dir():
        pytest.skip(f"prior {prior_id} is not staged under DMPO_PRIOR_ROOT")
    return str(path)


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


def _dmpo_normalizer(proprio_size: int = 3) -> obs_utils.DictRunningStatisticsState:
    def state(size):
        return running_statistics.init_state(specs.Array((size,), jnp.dtype("float32")))

    return obs_utils.DictRunningStatisticsState(
        imitation_target=state(2), proprioception=state(proprio_size)
    )


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
        _prior_path("260407_183223_056788")
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
        _prior_path("260414_072827_774853"), step=24
    )
    fn = prior_utils.make_prior_inference_fn(prior, norm, cfg)
    proprio = jnp.zeros((2, cfg["network_config"]["obs_sizes"]["proprioception"]))
    mean, logvar = fn(proprio)
    assert (
        mean.shape == (2, cfg["network_config"]["intention_size"])
        and logvar.shape == mean.shape
    )


@needs_prior_artifacts
def test_decoder_fn_logits_agree_with_local_logits_fn():
    """SCAMPER's decoder factory and the local logits factory see the same logits."""
    enc, prior, dec, norm, cfg = prior_utils.load_prior_checkpoint(
        _prior_path("260827_153154_576196"), step=24
    )
    nc = cfg["network_config"]
    rng = np.random.RandomState(0)
    latent_proprio = jnp.asarray(
        rng.randn(3, nc["intention_size"] + nc["obs_sizes"]["proprioception"]).astype(
            "float32"
        )
    )

    action, extras = prior_utils.make_decoder_inference_fn(dec, norm, cfg)(
        latent_proprio
    )
    logits, _ = prior_utils.make_decoder_logits_fn(dec, norm, cfg)(latent_proprio)

    assert action.shape == (3, nc["action_size"])
    assert logits.shape == (3, 2 * nc["action_size"])
    np.testing.assert_array_equal(np.asarray(extras["logits"]), np.asarray(logits))


@needs_prior_artifacts
@pytest.mark.parametrize("prior_id", PRIOR_IDS)
def test_loader_matches_vendored_checksums(prior_id):
    """Every restored leaf is bit-identical to what the vendored loader gave."""
    golden = json.loads(GOLDEN_PATH.read_text())["priors"][prior_id]
    path = _prior_path(prior_id)
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
    *_, norm, _cfg = prior_utils.load_prior_checkpoint(_prior_path(prior_id))
    assert isinstance(norm, obs_utils.DictRunningStatisticsState)


def test_prior_is_the_scamper_prior():
    assert prior_utils.Prior is prior_networks.Prior


def test_vendored_loader_helpers_are_gone():
    assert not hasattr(prior_utils, "_create_abstract_prior_policy")
    assert not hasattr(prior_utils, "_init_network_params")
    # Kept local: SCAMPER has no logits-only decoder factory.
    assert callable(prior_utils.make_decoder_logits_fn)


def test_loader_delegates_and_converts_the_normalizer(monkeypatch):
    """The SCAMPER loader's own state class becomes the DMPO one, values intact."""
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


def test_inference_factories_delegate_to_scamper(monkeypatch):
    norm = _dmpo_normalizer()

    # Unpatched: the returned callables are SCAMPER's closures, and they accept
    # DMPO's normalizer type. The logits factory is the one local implementation.
    decoder_fn = prior_utils.make_decoder_inference_fn({}, norm, TINY_CONFIG)
    prior_fn = prior_utils.make_prior_inference_fn({}, norm, TINY_CONFIG)
    logits_fn = prior_utils.make_decoder_logits_fn({}, norm, TINY_CONFIG)
    assert decoder_fn.__module__ == checkpoint_utils.__name__
    assert prior_fn.__module__ == checkpoint_utils.__name__
    assert logits_fn.__module__ == prior_utils.__name__

    # Patched: the public entry points call SCAMPER's functions and return what
    # they return, with the arguments untouched.
    for name in ("make_decoder_inference_fn", "make_prior_inference_fn"):
        sentinel = object()
        calls = []

        def fake(*args, _calls=calls, _sentinel=sentinel):
            _calls.append(args)
            return _sentinel

        monkeypatch.setattr(checkpoint_utils, name, fake)
        params = {"p": 1}
        assert getattr(prior_utils, name)(params, norm, TINY_CONFIG) is sentinel
        assert calls == [(params, norm, TINY_CONFIG)]


def test_missing_scamper_message(scamper_missing):
    install = re.escape("pip install --no-deps -r requirements-dmpo-prior.txt")
    pin = REQUIREMENTS_PATH.read_text().strip()
    norm = _dmpo_normalizer()
    needs_scamper = [
        lambda: prior_utils.load_prior_checkpoint("does/not/matter"),
        lambda: prior_utils.Prior,
        lambda: prior_utils.make_decoder_inference_fn({}, norm, TINY_CONFIG),
        lambda: prior_utils.make_prior_inference_fn({}, norm, TINY_CONFIG),
    ]
    for call in needs_scamper:
        with pytest.raises(ImportError, match=install) as err:
            call()
        # The message also carries the pinned requirement for installs without
        # the requirements file (it must match that file exactly).
        assert pin in str(err.value)

    # Only the decoder-logits factory is local, so it works without SCAMPER.
    assert callable(prior_utils.make_decoder_logits_fn({}, norm, TINY_CONFIG))
