"""Tests for RNN hidden state propagation with per-code reinit and z_e at action head.

Tests verify:
1. Hidden state continuity within a code (normal GRU propagation)
2. Hidden state reinit on code transition (zeros or learned)
3. Hidden state reset on episode done (always zeros, NOT learned init)
4. z_e at action head (GRU input excludes z_e, action head includes z_e)
5. apply_sequence matches manual step-by-step execution
6. Edge cases: done + code change, first step behavior, all-same-code

Run:
    cd moseq_jax
    python -m pytest tests/test_rnn_hidden_propagation.py -v
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure moseq_jax is importable
_MOSEQ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MOSEQ_DIR))

from moseq_decoder_network import MoSeqRecurrentDecoderNetwork

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CODES = 5
CODE_EMBED_DIM = 4
RNN_HIDDEN = (8,)  # single small GRU layer for fast tests
PROPRIO_DIM = 6
ACTION_DIM = 4
BATCH_SIZE = 2
CODE_STACK_SIZE = 1  # single code per step for clarity


def _make_obs(codes: np.ndarray, batch_size: int = BATCH_SIZE) -> dict:
    """Create a fake obs dict for a single timestep.

    Args:
        codes: [B] array of code indices.
    """
    return {
        "kpms_code": jnp.array(codes, dtype=jnp.float32).reshape(batch_size, 1),
        "proprioception": jnp.ones((batch_size, PROPRIO_DIM)),
    }


def _make_obs_seq(code_seq: list[list[int]], T: int, B: int) -> dict:
    """Create obs sequence [T, B, ...].

    Args:
        code_seq: [T][B] list of code indices per step per batch.
    """
    codes = jnp.array(code_seq, dtype=jnp.float32).reshape(T, B, 1)
    proprio = jnp.ones((T, B, PROPRIO_DIM))
    return {"kpms_code": codes, "proprioception": proprio}


def _init_network(
    reinit_hidden_on_code: bool = False,
    learned_hidden_init: bool = False,
    z_e_at_action_head: bool = False,
    use_continuous_encoder: bool = False,
):
    """Initialize a small test network and return (module, params, init_hidden)."""
    module = MoSeqRecurrentDecoderNetwork(
        num_codes=NUM_CODES,
        code_embed_dim=CODE_EMBED_DIM,
        rnn_hidden_sizes=RNN_HIDDEN,
        action_param_size=ACTION_DIM * 2,
        use_continuous_encoder=use_continuous_encoder,
        continuous_latent_dim=2,
        z_e_at_action_head=z_e_at_action_head,
        reinit_hidden_on_code=reinit_hidden_on_code,
        learned_hidden_init=learned_hidden_init,
    )

    key = jax.random.PRNGKey(0)
    dummy_obs = {
        "kpms_code": jnp.zeros((BATCH_SIZE, 1)),
        "proprioception": jnp.zeros((BATCH_SIZE, PROPRIO_DIM)),
    }
    if use_continuous_encoder:
        dummy_obs["task_obs"] = jnp.zeros((BATCH_SIZE, 10))
    dummy_hidden = [jnp.zeros((BATCH_SIZE, h)) for h in RNN_HIDDEN]

    params = module.init(key, dummy_obs, dummy_hidden, key)
    init_hidden = [jnp.zeros((BATCH_SIZE, h)) for h in RNN_HIDDEN]
    return module, params, init_hidden


# ---------------------------------------------------------------------------
# Test 1: No reinit — hidden propagates normally
# ---------------------------------------------------------------------------


def test_no_reinit_hidden_propagates():
    """Without reinit, hidden state flows continuously through steps."""
    module, params, init_hidden = _init_network(reinit_hidden_on_code=False)

    # Same code for all steps — no transitions
    T = 4
    code_seq = [[1, 2]] * T  # [T, B]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.zeros((T, BATCH_SIZE))
    key = jax.random.PRNGKey(42)

    # Run apply_sequence
    logits_seq, _, _, final_hidden = module.apply(
        params, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # Run step-by-step manually
    hidden = [h.copy() for h in init_hidden]
    manual_logits = []
    for t in range(T):
        obs_t = {k: v[t] for k, v in obs_seq.items()}
        action_params, code_idx, mean, logvar, new_hidden = module.apply(
            params, obs_t, hidden, key, deterministic=True, z_e_scale=1.0,
        )
        manual_logits.append(action_params)
        hidden = new_hidden

    manual_logits = jnp.stack(manual_logits)

    # Actions should match (relaxed tol: scan vs manual stepping has minor float accumulation diffs)
    np.testing.assert_allclose(logits_seq, manual_logits, atol=1e-3,
                                err_msg="apply_sequence should match manual stepping without reinit")
    # Final hidden should match
    for h_seq, h_manual in zip(final_hidden, hidden):
        np.testing.assert_allclose(h_seq, h_manual, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 2: Zero reinit on code transition
# ---------------------------------------------------------------------------


def test_zero_reinit_on_code_transition():
    """With reinit (zeros), hidden resets to 0 when code changes."""
    module, params, init_hidden = _init_network(
        reinit_hidden_on_code=True, learned_hidden_init=False,
    )

    T = 6
    # Code changes at step 3: [1,1,1,2,2,2]
    code_seq = [[1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2]]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.zeros((T, BATCH_SIZE))
    key = jax.random.PRNGKey(42)

    logits_seq, _, _, _ = module.apply(
        params, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # Step 0 and step 3 should both start from zeros (reinit at code change)
    # So: logits at step 0 (code=1, h=0) and step 3 (code=2, h=0) both use
    # freshly zeroed hidden. The actions differ only because the code embedding differs.

    # More importantly: step 3 action should be DIFFERENT from step 2 action
    # because the hidden was reset (not carried from step 2)
    assert not jnp.allclose(logits_seq[2], logits_seq[3], atol=1e-3), \
        "Step 3 (after reinit) should differ from step 2 (before reinit)"

    # Steps 4 and 5 should show GRU propagation from the zeroed hidden at step 3
    # (not from step 2's hidden)

    # Verify by running steps 3-5 manually from zero hidden with code=2
    hidden = [jnp.zeros((BATCH_SIZE, h)) for h in RNN_HIDDEN]
    for t in range(3, T):
        obs_t = {k: v[t] for k, v in obs_seq.items()}
        action_params, _, _, _, new_hidden = module.apply(
            params, obs_t, hidden, key, deterministic=True,
        )
        if t == 3:
            np.testing.assert_allclose(action_params, logits_seq[3], atol=1e-5,
                                        err_msg="Step 3 should match fresh-hidden forward pass")
        hidden = new_hidden


# ---------------------------------------------------------------------------
# Test 3: Learned reinit on code transition
# ---------------------------------------------------------------------------


def test_learned_reinit_on_code_transition():
    """With learned reinit, hidden resets to learned per-code vectors."""
    module, params, init_hidden = _init_network(
        reinit_hidden_on_code=True, learned_hidden_init=True,
    )

    # Check that learned init params exist and have correct shape
    param_tree = jax.tree_util.tree_map(lambda x: x.shape, params)
    # Should have hidden_init_0 with shape [NUM_CODES, RNN_HIDDEN[0]]
    assert "params" in params
    found_init = False
    for key_path, leaf in jax.tree_util.tree_leaves_with_path(params):
        path_str = "/".join(str(k) for k in key_path)
        if "hidden_init" in path_str:
            found_init = True
            assert leaf.shape == (NUM_CODES, RNN_HIDDEN[0]), \
                f"hidden_init shape should be ({NUM_CODES}, {RNN_HIDDEN[0]}), got {leaf.shape}"
    assert found_init, "Should have hidden_init_0 params"

    # Make learned inits non-zero for testing
    def set_inits(params):
        """Mutate learned init params to be non-zero for testing."""
        flat, treedef = jax.tree_util.tree_flatten_with_path(params)
        new_leaves = []
        for path, leaf in flat:
            path_str = "/".join(str(k) for k in path)
            if "hidden_init" in path_str:
                # Set each code's init to a distinct non-zero value
                new_leaf = jnp.arange(leaf.size, dtype=jnp.float32).reshape(leaf.shape) * 0.1
                new_leaves.append(new_leaf)
            else:
                new_leaves.append(leaf)
        return treedef.unflatten(new_leaves)

    params = set_inits(params)

    T = 4
    # Code changes at step 2: [1,1,3,3]
    code_seq = [[1, 1], [1, 1], [3, 3], [3, 3]]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.zeros((T, BATCH_SIZE))
    key = jax.random.PRNGKey(42)

    logits_seq, _, _, _ = module.apply(
        params, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # Step 2 should use learned init for code 3 (not zeros, not carried hidden)
    # Verify by running step 2 manually with code 3's learned init
    learned_inits = None
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        path_str = "/".join(str(k) for k in path)
        if "hidden_init_0" in path_str:
            learned_inits = leaf
            break

    assert learned_inits is not None
    code3_init = learned_inits[3]  # [H]
    hidden_for_code3 = [code3_init[None, :].repeat(BATCH_SIZE, axis=0)]  # [B, H]

    obs_step2 = {k: v[2] for k, v in obs_seq.items()}
    action_step2, _, _, _, _ = module.apply(
        params, obs_step2, hidden_for_code3, key, deterministic=True,
    )
    np.testing.assert_allclose(action_step2, logits_seq[2], atol=1e-3,
                                err_msg="Step 2 should use learned init for code 3")


# ---------------------------------------------------------------------------
# Test 4: Episode done resets to zeros (NOT learned init)
# ---------------------------------------------------------------------------


def test_done_resets_to_zeros_not_learned():
    """Episode done should reset hidden to zeros, even with learned init enabled."""
    module, params, init_hidden = _init_network(
        reinit_hidden_on_code=True, learned_hidden_init=True,
    )

    T = 4
    # Same code throughout, done at step 1
    code_seq = [[2, 2], [2, 2], [2, 2], [2, 2]]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.array([[0, 0], [1, 1], [0, 0], [0, 0]], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    logits_seq, _, _, _ = module.apply(
        params, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # After done at step 1, hidden is reset to ZEROS (not learned init)
    # Step 2 should use zero hidden (code hasn't changed from code 2,
    # but the hidden was zeroed by done at step 1)
    #
    # However, there's a subtlety: at step 2, prev_code = 2 (from step 1),
    # code_t = 2, so code_changed = False. The reinit does NOT fire.
    # The hidden IS zeros (from done reset at step 1), but that's from
    # _reset_hidden_on_done, not _reinit_hidden_on_code.
    #
    # Verify: step 2 output should match running with zero hidden + code=2
    zero_hidden = [jnp.zeros((BATCH_SIZE, h)) for h in RNN_HIDDEN]
    obs_step2 = {k: v[2] for k, v in obs_seq.items()}
    action_step2, _, _, _, _ = module.apply(
        params, obs_step2, zero_hidden, key, deterministic=True,
    )
    np.testing.assert_allclose(action_step2, logits_seq[2], atol=1e-5,
                                err_msg="After done, step 2 should use zero hidden")


# ---------------------------------------------------------------------------
# Test 5: z_e at action head — GRU input excludes z_e
# ---------------------------------------------------------------------------


def test_z_e_at_action_head():
    """With z_e_at_action_head, GRU dynamics should be identical with/without z_e."""
    module_with_ze, params_with_ze, init_hidden = _init_network(
        z_e_at_action_head=True, use_continuous_encoder=True,
    )

    T = 3
    code_seq = [[1, 2]] * T
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    # Add task_obs for the continuous encoder
    obs_seq["task_obs"] = jnp.ones((T, BATCH_SIZE, 10))
    done_seq = jnp.zeros((T, BATCH_SIZE))
    key = jax.random.PRNGKey(42)

    # Run with z_e_scale=1.0 (full z_e at action head)
    logits_full, _, _, hidden_full = module_with_ze.apply(
        params_with_ze, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module_with_ze.apply_sequence,
    )

    # Run with z_e_scale=0.0 (no z_e at action head)
    logits_zero, _, _, hidden_zero = module_with_ze.apply(
        params_with_ze, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=0.0,
        method=module_with_ze.apply_sequence,
    )

    # GRU hidden states should be IDENTICAL (z_e doesn't enter GRU)
    for h_full, h_zero in zip(hidden_full, hidden_zero):
        np.testing.assert_allclose(h_full, h_zero, atol=1e-5,
                                    err_msg="GRU hidden should be identical with/without z_e when z_e_at_action_head=True")

    # Action outputs should DIFFER (z_e enters action head)
    assert not jnp.allclose(logits_full, logits_zero, atol=1e-3), \
        "Actions should differ between z_e=1 and z_e=0 (z_e enters action head)"


# ---------------------------------------------------------------------------
# Test 6: First step always reinits (prev_code starts at -1)
# ---------------------------------------------------------------------------


def test_first_step_always_reinits():
    """At t=0, prev_code=-1, so code_changed=True and reinit fires."""
    module, params, _ = _init_network(
        reinit_hidden_on_code=True, learned_hidden_init=False,
    )

    T = 2
    code_seq = [[1, 1], [1, 1]]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.zeros((T, BATCH_SIZE))
    key = jax.random.PRNGKey(42)

    # Pass non-zero initial hidden
    nonzero_hidden = [jnp.ones((BATCH_SIZE, h)) * 5.0 for h in RNN_HIDDEN]

    logits_seq, _, _, _ = module.apply(
        params, obs_seq, nonzero_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # Step 0 should have ZEROED hidden (reinit fires because prev_code=-1)
    # NOT the nonzero_hidden we passed in
    zero_hidden = [jnp.zeros((BATCH_SIZE, h)) for h in RNN_HIDDEN]
    obs_step0 = {k: v[0] for k, v in obs_seq.items()}
    action_from_zero, _, _, _, _ = module.apply(
        params, obs_step0, zero_hidden, key, deterministic=True,
    )
    np.testing.assert_allclose(action_from_zero, logits_seq[0], atol=1e-5,
                                err_msg="Step 0 should use zeroed hidden (reinit from prev_code=-1)")


# ---------------------------------------------------------------------------
# Test 7: Code transition + done at same step
# ---------------------------------------------------------------------------


def test_code_transition_and_done_same_step():
    """When both code changes AND done=True at same step, done-reset wins for NEXT step."""
    module, params, init_hidden = _init_network(
        reinit_hidden_on_code=True, learned_hidden_init=False,
    )

    T = 4
    # Code changes at step 2 AND done at step 2
    code_seq = [[1, 1], [1, 1], [3, 3], [3, 3]]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.array([[0, 0], [0, 0], [1, 1], [0, 0]], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    logits_seq, _, _, _ = module.apply(
        params, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # At step 2: code_changed=True (1→3), so hidden reinit to zeros BEFORE forward pass
    # Then done=True, so hidden reset to zeros AFTER forward pass
    # At step 3: prev_code=3, code_t=3, code_changed=False
    # Hidden is zeros (from done reset at step 2)
    # So step 3 should use zero hidden + code=3

    zero_hidden = [jnp.zeros((BATCH_SIZE, h)) for h in RNN_HIDDEN]
    obs_step3 = {k: v[3] for k, v in obs_seq.items()}
    action_step3, _, _, _, _ = module.apply(
        params, obs_step3, zero_hidden, key, deterministic=True,
    )
    np.testing.assert_allclose(action_step3, logits_seq[3], atol=1e-5,
                                err_msg="Step 3 after done+code_change should use zero hidden")


# ---------------------------------------------------------------------------
# Test 8: Different codes produce different learned inits
# ---------------------------------------------------------------------------


def test_different_codes_different_inits():
    """Each code should have a distinct learned initial hidden state."""
    module, params, init_hidden = _init_network(
        reinit_hidden_on_code=True, learned_hidden_init=True,
    )

    # Set distinct inits per code
    def set_distinct_inits(params):
        flat, treedef = jax.tree_util.tree_flatten_with_path(params)
        new_leaves = []
        for path, leaf in flat:
            path_str = "/".join(str(k) for k in path)
            if "hidden_init" in path_str:
                # Each code gets a unique init
                new_leaf = jnp.stack([
                    jnp.full(leaf.shape[1:], float(c) * 0.5)
                    for c in range(leaf.shape[0])
                ])
                new_leaves.append(new_leaf)
            else:
                new_leaves.append(leaf)
        return treedef.unflatten(new_leaves)

    params = set_distinct_inits(params)

    T = 2
    key = jax.random.PRNGKey(42)

    # Run with code=1 for batch element 0, code=3 for batch element 1
    code_seq = [[1, 3], [1, 3]]
    obs_seq = _make_obs_seq(code_seq, T, BATCH_SIZE)
    done_seq = jnp.zeros((T, BATCH_SIZE))

    logits_seq, _, _, _ = module.apply(
        params, obs_seq, init_hidden, done_seq, key,
        deterministic=True, z_e_scale=1.0,
        method=module.apply_sequence,
    )

    # Batch element 0 (code=1) and batch element 1 (code=3) should produce
    # different actions at step 0 because they start from different learned inits
    assert not jnp.allclose(logits_seq[0, 0], logits_seq[0, 1], atol=1e-3), \
        "Different codes should produce different actions due to different learned inits"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
