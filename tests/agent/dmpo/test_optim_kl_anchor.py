import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.optim_kl_anchor import (
    make_kl_anchor_optimizers,
    label_param_tree,
)


def test_label_param_tree_classifies_blocks():
    params = {
        "params": {
            "prior": {"hidden_0": {"kernel": jnp.zeros((4, 4))}},
            "decoder": {"hidden_0": {"kernel": jnp.zeros((4, 4))}},
            "policy_head": {"hidden_0": {"kernel": jnp.zeros((4, 4))}},
        }
    }
    labels = label_param_tree(params)
    leaves, _ = jax.tree_util.tree_flatten(labels)
    assert "prior" in leaves
    assert "decoder" in leaves
    assert "policy_head" in leaves


def test_optimizer_applies_different_lr_per_block():
    cfg = DMPOConfig(
        num_envs=4,
        unroll_length=4,
        batch_size=4,
        sequence_length=4,
        policy_lr=1.0,  # large lr to make the difference visible
    )
    pol_opt, _, _ = make_kl_anchor_optimizers(
        cfg, prior_lr_mult=0.1, decoder_lr_mult=1.0
    )

    params = {
        "params": {
            "prior": {"k": jnp.zeros((1,))},
            "decoder": {"k": jnp.zeros((1,))},
            "policy_head": {"k": jnp.zeros((1,))},
        }
    }
    grads = {
        "params": {
            "prior": {"k": jnp.ones((1,))},
            "decoder": {"k": jnp.ones((1,))},
            "policy_head": {"k": jnp.ones((1,))},
        }
    }
    state = pol_opt.init(params)
    updates, _ = pol_opt.update(grads, state, params)
    prior_step = float(updates["params"]["prior"]["k"][0])
    decoder_step = float(updates["params"]["decoder"]["k"][0])
    policy_step = float(updates["params"]["policy_head"]["k"][0])
    assert prior_step < 0 and decoder_step < 0 and policy_step < 0
    assert abs(prior_step) < abs(decoder_step) * 0.5, (
        f"prior step {prior_step} not noticeably smaller than decoder step {decoder_step}"
    )
