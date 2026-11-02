"""Loader tests against real SCAMPER mlp_prior checkpoints on disk.

Skipped unless ``DMPO_PRIOR_ROOT`` points at a directory of mlp_prior
checkpoints (e.g. ``SCAMPER/model_checkpoints``) — these need real on-disk
artifacts, not synthetic fixtures.
"""

import os

import pytest

from track_mjx.agent.dmpo import prior_utils

ROOT = os.environ.get("DMPO_PRIOR_ROOT")
# Module-level marker is fine here: every test in this file needs the
# on-disk prior artifacts, so there is no mixed-file case to worry about.
pytestmark = [
    pytest.mark.artifact,
    pytest.mark.skipif(
        ROOT is None, reason="DMPO_PRIOR_ROOT not set (prior artifacts absent)"
    ),
]


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


def test_prior_fn_returns_latent_mean_and_logvar():
    enc, prior, dec, norm, cfg = prior_utils.load_prior_checkpoint(
        f"{ROOT}/260414_072827_774853", step=24
    )
    fn = prior_utils.make_prior_inference_fn(prior, norm, cfg)
    import jax.numpy as jnp

    proprio = jnp.zeros((2, cfg["network_config"]["obs_sizes"]["proprioception"]))
    mean, logvar = fn(proprio)
    assert (
        mean.shape == (2, cfg["network_config"]["intention_size"])
        and logvar.shape == mean.shape
    )
