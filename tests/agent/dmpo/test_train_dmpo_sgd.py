"""Tests that scan_k_sgd produces the same final state as K sequential
sgd_step calls (modulo replay sampling, which we sidestep by stubbing
the buffer to deterministically return the same fixed batch every call).
"""
import jax
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("K", [1, 2, 4])
def test_scan_k_sgd_advances_state_steps_by_K(K):
    """state.steps should advance by K after one scan_k_sgd call."""
    from track_mjx.agent.dmpo.train_dmpo_sgd import make_scan_k_sgd
    from track_mjx.agent.dmpo.config import DMPOConfig
    from track_mjx.agent.dmpo.networks import make_dmpo_networks
    from track_mjx.agent.dmpo.learner import init_training_state, make_optimizers

    cfg = DMPOConfig(num_envs=4, batch_size=8, sequence_length=4, unroll_length=4)
    nets = make_dmpo_networks(obs_size=6, action_size=3, cfg=cfg)
    optimizers = make_optimizers(cfg)
    rng = jax.random.PRNGKey(0)
    env_spec = {"obs_size": 6, "action_size": 3}
    state = init_training_state(rng, nets, env_spec, cfg)

    # Build a stub batch of the shape sgd_step expects.
    fake_batch = {
        "observation": jnp.zeros((cfg.batch_size, cfg.sequence_length, 6)),
        "action": jnp.zeros((cfg.batch_size, cfg.sequence_length, 3)),
        "reward": jnp.zeros((cfg.batch_size, cfg.sequence_length)),
        "discount": jnp.ones((cfg.batch_size, cfg.sequence_length)),
        "next_observation": jnp.zeros((cfg.batch_size, cfg.sequence_length, 6)),
    }

    # Stub buffer: sample always returns the same fake batch.
    class _StubBuf:
        def sample(self, _state, _key):
            class _S: pass
            s = _S()
            s.experience = fake_batch
            return s

    rb = _StubBuf()
    scan_k_sgd = make_scan_k_sgd(rb, nets, optimizers, cfg, K=K)
    new_state, metrics = scan_k_sgd(state, rb_state=None, rng=rng)
    assert int(new_state.steps) == int(state.steps) + K
    # metrics dict should still be flat scalars (averaged across K).
    assert metrics["critic_loss"].shape == ()


def test_scan_k_sgd_K1_matches_raw_sgd_step():
    """K=1 scan_k_sgd should produce a final state pytree-equal to a single sgd_step."""
    from track_mjx.agent.dmpo.train_dmpo_sgd import make_scan_k_sgd
    from track_mjx.agent.dmpo.learner import sgd_step, init_training_state, make_optimizers
    from track_mjx.agent.dmpo.networks import make_dmpo_networks
    from track_mjx.agent.dmpo.config import DMPOConfig

    cfg = DMPOConfig(num_envs=4, batch_size=8, sequence_length=4, unroll_length=4)
    nets = make_dmpo_networks(obs_size=6, action_size=3, cfg=cfg)
    optimizers = make_optimizers(cfg)
    rng = jax.random.PRNGKey(42)
    env_spec = {"obs_size": 6, "action_size": 3}
    state = init_training_state(rng, nets, env_spec, cfg)

    fake_batch = {
        "observation": jnp.zeros((cfg.batch_size, cfg.sequence_length, 6)),
        "action": jnp.zeros((cfg.batch_size, cfg.sequence_length, 3)),
        "reward": jnp.zeros((cfg.batch_size, cfg.sequence_length)),
        "discount": jnp.ones((cfg.batch_size, cfg.sequence_length)),
        "next_observation": jnp.zeros((cfg.batch_size, cfg.sequence_length, 6)),
    }

    class _StubBuf:
        def sample(self, _s, _k):
            class _S: pass
            s = _S(); s.experience = fake_batch; return s

    raw_state, _ = sgd_step(state, fake_batch, nets, optimizers, cfg)
    scan_state, _ = make_scan_k_sgd(_StubBuf(), nets, optimizers, cfg, K=1)(state, None, rng)

    leaves_raw = jax.tree.leaves(raw_state.policy_params)
    leaves_scan = jax.tree.leaves(scan_state.policy_params)
    for a, b in zip(leaves_raw, leaves_scan):
        assert jnp.allclose(a, b, atol=1e-6), "K=1 scan diverged from raw sgd_step"
