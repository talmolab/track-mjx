"""End-to-end fused-step / chunk / training-loop wiring for the recurrent head.

Unlike test_rollout_rnn / test_learner_rnn (which stub the policy to pin the
per-module contracts), this suite uses the REAL recurrent networks
(``make_dmpo_kl_anchor_rnn_networks``) at tiny sizes against a stub
pre_batched dict-obs env, so it exercises the actual cross-module seams the
Core stage could not: the 5-arg fused step signature, the hidden riding the
chunk's scan carry, the replay schema with ``policy_hidden``, and the
``sgd_step -> _sgd_step_rnn`` dispatch on real sampled batches. vision 8x8 is
the floor for cnn_channels=(2,4,8) with the VisionEncoder default strides
(1,1,2): 8 ->conv3x3-> 6 -> 4 ->s2-> 1 (see test_networks_kl_anchor_rnn).

Coverage:
  (a) fused step runs twice — reset signature (env_state=None,
      policy_hidden=None) then resume signature — and THREADS the hidden
      (call 2's first stored pre-step hidden == call 1's final hidden);
  (b) a 2-iter chunk runs with the hidden in the scan carry;
  (c) the replay state contains ``policy_hidden`` in store_dtype;
  (d) metrics contain the rnn/ diagnostics;
  (e) the FF fused build from the SAME module still works (4-arg signature,
      no policy_hidden anywhere) — the wiring did not disturb the FF path;
  (f) training_loop.run threads the hidden through warmup + chunk.
"""
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    TrainingState,
    _build_loss,
    make_optimizers,
)
from track_mjx.agent.dmpo.networks_kl_anchor_rnn import (
    make_dmpo_kl_anchor_rnn_networks,
)
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.train_dmpo_chunk import make_train_chunk
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step
from track_mjx.agent.dmpo.training_loop import run as train_loop_run
from track_mjx.agent.observation_utils import init_dict_normalizer

from tests.agent.dmpo.test_train_dmpo_fused import _setup as _setup_ff

VISION_SHAPE = (8, 8, 2)
TASK_OBS = 4
PROPRIO = 5
ACTION = 4
LATENT = 3
RNN_HIDDEN = (6,)


@flax.struct.dataclass
class _State:
    obs: dict
    t: jax.Array
    done: jax.Array
    reward: jax.Array


def _obs_at(t):
    """Deterministic per-env, per-step dict obs; vision stays in [0, 1]."""
    n = t.shape[0]
    phase = (0.1 * t.astype(jnp.float32)
             + 0.01 * jnp.arange(n, dtype=jnp.float32))[:, None]
    grid = jnp.linspace(0.0, 1.0, int(np.prod(VISION_SHAPE)), dtype=jnp.float32)
    vision = jnp.clip(0.5 * grid[None, :] + 0.25 * jnp.sin(phase), 0.0, 1.0)
    return {
        "vision": vision.reshape((n,) + VISION_SHAPE),
        "imitation_target": phase + 0.001 * jnp.arange(TASK_OBS, dtype=jnp.float32)[None, :],
        "proprioception": jnp.cos(phase + 0.1 * jnp.arange(PROPRIO, dtype=jnp.float32)[None, :]),
    }


class _DictObsEnv:
    """Pre-batched counting env emitting the production obs-dict schema."""

    pre_batched = True
    action_size = ACTION

    def reset(self, keys):
        n = keys.shape[0]
        t = jnp.zeros((n,), jnp.int32)
        return _State(obs=_obs_at(t), t=t,
                      done=jnp.zeros((n,)), reward=jnp.zeros((n,)))

    def step(self, st, action):
        t = st.t + 1
        reward = jnp.tanh(jnp.linalg.norm(action, axis=-1))
        return _State(obs=_obs_at(t), t=t,
                      done=jnp.zeros_like(st.done), reward=reward), reward


def _obs_template():
    return {
        "vision": jnp.zeros(VISION_SHAPE, jnp.float32),
        "imitation_target": jnp.zeros((TASK_OBS,), jnp.float32),
        "proprioception": jnp.zeros((PROPRIO,), jnp.float32),
    }


def _setup_rnn(seed=0):
    """Real recurrent nets + TrainingState + replay against the stub env.

    The state is built by hand (as in test_learner_rnn) rather than via
    init_training_state, whose ``policy.init(rng, obs)`` call is FF-shaped —
    the entry point adapts it with a default-hidden wrapper; here the direct
    3-arg init is the simpler equivalent.
    """
    cfg = DMPOConfig(
        num_envs=2,
        batch_size=4,
        # T = rnn_bptt_length + n_step, checked fail-loud by the learner.
        sequence_length=5,
        n_step=3,
        rnn_bptt_length=2,
        use_n_step=True,
        store_next_observation=False,
        # unroll 8 > min_size 6 (= seq+1) so can_sample flips True at the end
        # of the first fused step — same reasoning as test_train_dmpo_fused.
        unroll_length=8,
        max_replay_size=64,
        min_replay_size=4,
        num_samples=3,
        num_atoms=11,
        vmin=0.0,
        vmax=5.0,
        discount=0.9,
    )
    env = _DictObsEnv()
    nets = make_dmpo_kl_anchor_rnn_networks(
        proprio_size=PROPRIO,
        task_obs_size=TASK_OBS,
        action_size=ACTION,
        latent_size=LATENT,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=(8,),
        decoder_layer_sizes=(8,),
        rnn_cell="gru",
        rnn_mlp_layers=(8,),
        rnn_hidden_sizes=RNN_HIDDEN,
        rnn_store_dtype="float16",
        cnn_feature_size=4,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        value_hidden_layer_sizes=(16,),
        critic_use_proprio=True,
    )
    optimizers = make_optimizers(cfg)
    pol_opt, crit_opt, dual_opt = optimizers

    rng = jax.random.PRNGKey(seed)
    rng, k_pol, k_crit = jax.random.split(rng, 3)
    obs_dummy = _obs_template()
    policy_params = nets.policy.init(
        k_pol, obs_dummy, nets.recurrent_meta.init_hidden()
    )
    critic_params = nets.critic.init(
        k_crit, obs_dummy, jnp.zeros((ACTION,), jnp.float32)
    )
    dual_params = _build_loss(cfg).init_params(ACTION, jnp.float32)
    state = TrainingState(
        policy_params=policy_params,
        critic_params=critic_params,
        target_policy_params=policy_params,
        target_critic_params=critic_params,
        dual_params=dual_params,
        policy_opt_state=pol_opt.init(policy_params),
        critic_opt_state=crit_opt.init(critic_params),
        dual_opt_state=dual_opt.init(dual_params),
        normalizer_params=init_dict_normalizer(obs_dummy),
        steps=jnp.zeros((), jnp.int32),
        rng=rng,
    )

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    # Compressed recurrent schema: no next_observation; policy_hidden leaves
    # are unbatched [H_l] in store_dtype — the entry point builds the same.
    transition_template = {
        "observation": obs_dummy,
        "action": jnp.zeros((ACTION,), jnp.float32),
        "reward": jnp.zeros((), jnp.float32),
        "discount": jnp.zeros((), jnp.float32),
        "policy_hidden": jax.tree.map(
            lambda h: h.astype(nets.recurrent_meta.store_dtype),
            nets.recurrent_meta.init_hidden(),
        ),
    }
    rb_state = rb.init(transition_template)

    return dict(
        cfg=cfg, env=env, nets=nets, optimizers=optimizers,
        state=state, rb=rb, rb_state=rb_state,
    )


def _assert_finite_scalars(metrics):
    for k, v in metrics.items():
        arr = np.asarray(v)
        assert arr.shape == (), f"metric {k} not scalar: {arr.shape}"
        assert np.isfinite(arr).all(), f"non-finite metric {k}: {arr}"


def test_fused_step_rnn_reset_then_resume_threads_hidden():
    """(a) + (c) + (d): two fused calls (reset / resume signatures); the
    hidden out of call 1 is the hidden call 2's rollout consumes at its first
    step (checked against what call 2 STORED in replay)."""
    s = _setup_rnn()
    cfg = s["cfg"]
    K = 1
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K,
    )

    # Reset signature: both env_state and policy_hidden None.
    state1, env_state1, hidden1, rb_state1, metrics1 = fused(
        s["state"], None, None, s["rb_state"], jax.random.PRNGKey(1),
    )
    assert isinstance(hidden1, tuple)
    for h, hl in zip(hidden1, RNN_HIDDEN):
        assert h.shape == (cfg.num_envs, hl)
        assert h.dtype == jnp.float32  # live hidden stays compute dtype
    hidden1_np = [np.asarray(h) for h in hidden1]
    assert any(np.any(h != 0.0) for h in hidden1_np), (
        "final hidden is all zeros after 8 steps -- the GRU never ran"
    )
    assert int(state1.steps) == K

    # (d) rnn/ diagnostics from _sgd_step_rnn, plus everything finite scalar.
    assert "rnn/hidden_staleness" in metrics1
    assert "rnn/hidden_abs_mean" in metrics1
    _assert_finite_scalars(metrics1)

    # (c) replay schema: policy_hidden present, stored in store_dtype.
    exp1 = rb_state1.experience
    assert "policy_hidden" in exp1
    assert isinstance(exp1["policy_hidden"], tuple)
    for leaf in exp1["policy_hidden"]:
        assert leaf.dtype == jnp.dtype("float16")
    write_off = int(rb_state1.current_index)
    assert write_off == cfg.unroll_length

    # Resume signature: concrete env_state + hidden. Threading proof: the
    # PRE-step hidden call 2 stores at its first step must equal call 1's
    # final hidden (up to the f16 storage cast) — anything else means the
    # carry was dropped or re-zeroed between fused calls.
    state2, env_state2, hidden2, rb_state2, metrics2 = fused(
        state1, env_state1, hidden1, rb_state1, jax.random.PRNGKey(2),
    )
    exp2 = rb_state2.experience
    for leaf, h1 in zip(exp2["policy_hidden"], hidden1_np):
        np.testing.assert_array_equal(
            np.asarray(leaf[:, write_off]),
            h1.astype(np.float16),
            err_msg="call 2's first stored pre-step hidden != call 1's final "
                    "hidden -- policy_hidden is not threading through the "
                    "fused-step carry",
        )
    for h_new, h_old in zip(hidden2, hidden1_np):
        assert not np.allclose(np.asarray(h_new), h_old), (
            "hidden did not evolve across the resumed rollout"
        )
    assert int(state2.steps) == 2 * K
    _assert_finite_scalars(metrics2)


def test_chunk_rnn_two_iters():
    """(b): a 2-iter recurrent chunk runs with the hidden in the scan carry
    and advances state / hidden / replay accordingly."""
    s = _setup_rnn()
    cfg = s["cfg"]
    K = 1
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K,
    )
    # Warm up so the chunk starts from concrete env_state + hidden (the
    # training loop's contract: warmup runs on fused_step first).
    state1, env_state1, hidden1, rb_state1, _ = fused(
        s["state"], None, None, s["rb_state"], jax.random.PRNGKey(3),
    )
    hidden1_np = [np.asarray(h) for h in hidden1]

    chunk = make_train_chunk(fused, n_iters=2, recurrent=True)
    state2, env_state2, hidden2, rb_state2, metrics = chunk(
        state1, env_state1, hidden1, rb_state1, jax.random.PRNGKey(4),
    )
    assert int(state2.steps) == K + 2 * K
    for h, hl in zip(hidden2, RNN_HIDDEN):
        assert h.shape == (cfg.num_envs, hl)
    for h_new, h_old in zip(hidden2, hidden1_np):
        assert not np.allclose(np.asarray(h_new), h_old)
    assert int(rb_state2.current_index) == 3 * cfg.unroll_length
    assert "rnn/hidden_staleness" in metrics
    _assert_finite_scalars(metrics)


def test_training_loop_threads_hidden_recurrent():
    """(f): run() detects nets.recurrent_meta and threads the hidden through
    the warmup fused step and the chunk without any caller-side plumbing."""
    s = _setup_rnn()
    cfg = s["cfg"]
    final_state, final_env_state, final_rb_state, metrics = train_loop_run(
        env=s["env"], nets=s["nets"], optimizers=s["optimizers"], rb=s["rb"],
        cfg=cfg, K=1, iters_per_chunk=2, rng=jax.random.PRNGKey(5),
        state=s["state"], env_state=None, rb_state=s["rb_state"],
        max_chunks=1,
    )
    # 1 warmup fused step + one 2-iter chunk, K=1 each.
    assert int(final_state.steps) == 3
    assert int(final_rb_state.current_index) == 3 * cfg.unroll_length
    assert "rnn/hidden_staleness" in metrics
    _assert_finite_scalars(metrics)


def test_ff_fused_build_unchanged_from_same_module():
    """(e): with recurrent_meta=None the SAME make_fused_train_step /
    make_train_chunk builds keep the 4-arg FF contract and emit no
    policy_hidden anywhere (the byte-identity guard proper is the untouched
    FF suite; this pins the signature from this module's perspective)."""
    s = _setup_ff()
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], s["cfg"], K=2,
    )
    out = fused(s["state"], None, s["rb_state"], jax.random.PRNGKey(0))
    assert len(out) == 4
    state1, env_state1, rb_state1, metrics = out
    assert "policy_hidden" not in rb_state1.experience
    assert not any(k.startswith("rnn/") for k in metrics)

    chunk = make_train_chunk(fused, n_iters=2)  # recurrent defaults OFF
    out2 = chunk(state1, env_state1, rb_state1, jax.random.PRNGKey(1))
    assert len(out2) == 4
