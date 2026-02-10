"""Tests for dead code reinitialization (Fix 3).

Verifies:
- Dead codes are correctly identified by usage threshold
- Active codes are preserved unchanged
- Param tree structure is maintained after reinit
- Reinitialized codes come from z_e samples
"""

import sys

sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx/vqvae_jax")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze

from vq_losses import reinit_dead_codes


def _make_policy_params(codebooks, rvq_depth):
    """Build a minimal policy param dict with codebook structure."""
    quantizer = {}
    for d in range(rvq_depth):
        quantizer[f"codebooks_{d}"] = {"embeddings": codebooks[d]}
    return freeze({"params": {"quantizer": quantizer}})


@pytest.fixture
def setup_dead_codes():
    """Create params where codes 2-7 are dead (only codes 0,1 used)."""
    K, D, depth = 8, 4, 2
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    cb0 = jax.random.normal(k1, (K, D))
    cb1 = jax.random.normal(k2, (K, D))
    z_e = jax.random.normal(k3, (100, D))

    policy_params = _make_policy_params((cb0, cb1), depth)

    # All indices point to codes 0 or 1 (codes 2-7 are dead)
    all_indices = (
        np.random.choice([0, 1], size=100),
        np.random.choice([0, 1], size=100),
    )

    return policy_params, z_e, all_indices, K, depth, cb0, cb1


def test_reinit_identifies_dead_codes(setup_dead_codes):
    """Codes with 0 usage should be reinitialized."""
    policy_params, z_e, all_indices, K, depth, cb0, cb1 = setup_dead_codes

    new_params = reinit_dead_codes(
        policy_params=policy_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        threshold=0.01,
        rng=jax.random.PRNGKey(42),
    )

    # Dead codes (2-7) should have been changed
    old_cb0 = cb0
    new_cb0 = new_params["params"]["quantizer"]["codebooks_0"]["embeddings"]

    for code_idx in range(2, K):
        assert not jnp.allclose(old_cb0[code_idx], new_cb0[code_idx], atol=1e-6), (
            f"Dead code {code_idx} should have been reinitialized"
        )


def test_reinit_preserves_alive_codes(setup_dead_codes):
    """Active codes (0, 1) should not be modified."""
    policy_params, z_e, all_indices, K, depth, cb0, cb1 = setup_dead_codes

    new_params = reinit_dead_codes(
        policy_params=policy_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        threshold=0.01,
        rng=jax.random.PRNGKey(42),
    )

    new_cb0 = new_params["params"]["quantizer"]["codebooks_0"]["embeddings"]

    # Active codes 0 and 1 should be unchanged
    assert jnp.allclose(cb0[0], new_cb0[0]), "Active code 0 should be preserved"
    assert jnp.allclose(cb0[1], new_cb0[1]), "Active code 1 should be preserved"


def test_reinit_param_tree_structure(setup_dead_codes):
    """Param tree shape/dtype should be preserved after reinit."""
    policy_params, z_e, all_indices, K, depth, cb0, cb1 = setup_dead_codes

    new_params = reinit_dead_codes(
        policy_params=policy_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        threshold=0.01,
        rng=jax.random.PRNGKey(42),
    )

    # Check structure is preserved
    for d in range(depth):
        old_emb = policy_params["params"]["quantizer"][f"codebooks_{d}"]["embeddings"]
        new_emb = new_params["params"]["quantizer"][f"codebooks_{d}"]["embeddings"]
        assert old_emb.shape == new_emb.shape, (
            f"Shape mismatch at depth {d}: {old_emb.shape} vs {new_emb.shape}"
        )
        assert old_emb.dtype == new_emb.dtype, (
            f"Dtype mismatch at depth {d}: {old_emb.dtype} vs {new_emb.dtype}"
        )


def test_reinit_new_values_from_z_e(setup_dead_codes):
    """Reinitialized codes should be close to z_e samples (+ small noise)."""
    policy_params, z_e, all_indices, K, depth, cb0, cb1 = setup_dead_codes

    noise_scale = 0.01
    new_params = reinit_dead_codes(
        policy_params=policy_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        threshold=0.01,
        noise_scale=noise_scale,
        rng=jax.random.PRNGKey(42),
    )

    new_cb0 = new_params["params"]["quantizer"]["codebooks_0"]["embeddings"]

    # Each reinitialized code should be close to some z_e sample
    for code_idx in range(2, K):
        new_code = new_cb0[code_idx]
        # Find nearest z_e sample
        distances = jnp.sum((z_e - new_code) ** 2, axis=-1)
        min_dist = jnp.min(distances)
        # Distance should be small (just noise)
        assert min_dist < noise_scale * 10, (
            f"Code {code_idx} distance to nearest z_e = {min_dist:.4f}, "
            f"expected < {noise_scale * 10:.4f}"
        )


def test_no_reinit_when_all_codes_active():
    """When all codes are active, nothing should change."""
    K, D, depth = 4, 4, 1
    cb = jax.random.normal(jax.random.PRNGKey(0), (K, D))
    z_e = jax.random.normal(jax.random.PRNGKey(1), (100, D))
    policy_params = _make_policy_params((cb,), depth)

    # All codes used uniformly
    all_indices = (np.array([0, 1, 2, 3] * 25),)

    new_params = reinit_dead_codes(
        policy_params=policy_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        threshold=0.01,
        rng=jax.random.PRNGKey(42),
    )

    new_cb = new_params["params"]["quantizer"]["codebooks_0"]["embeddings"]
    assert jnp.allclose(cb, new_cb), "No codes should be reinitialized when all are active"


def test_reinit_preserves_dict_type():
    """reinit_dead_codes should preserve the input pytree type (dict vs FrozenDict).

    In Flax 0.12.2, Module.init() returns regular dict. If reinit_dead_codes
    converts to FrozenDict, the optimizer state tree structure will mismatch,
    causing ValueError: Custom node type mismatch.
    """
    from flax.core.frozen_dict import FrozenDict

    K, D, depth = 4, 4, 1
    cb = jax.random.normal(jax.random.PRNGKey(0), (K, D))
    z_e = jax.random.normal(jax.random.PRNGKey(1), (100, D))
    # All indices point to code 0 (codes 1-3 are dead)
    all_indices = (np.zeros(100, dtype=np.int32),)

    # Test with regular dict input (Flax 0.12.2 default)
    dict_params = {"params": {"quantizer": {"codebooks_0": {"embeddings": cb}}}}
    result_dict = reinit_dead_codes(
        policy_params=dict_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        rng=jax.random.PRNGKey(42),
    )
    assert type(result_dict) is dict, (
        f"Expected dict output for dict input, got {type(result_dict)}"
    )
    assert type(result_dict["params"]) is dict, (
        f"Nested dict should stay dict, got {type(result_dict['params'])}"
    )

    # Test with FrozenDict input (legacy Flax)
    frozen_params = freeze(dict_params)
    result_frozen = reinit_dead_codes(
        policy_params=frozen_params,
        z_e_samples=z_e,
        all_indices=all_indices,
        num_codes=K,
        rvq_depth=depth,
        rng=jax.random.PRNGKey(42),
    )
    assert isinstance(result_frozen, FrozenDict), (
        f"Expected FrozenDict output for FrozenDict input, got {type(result_frozen)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
