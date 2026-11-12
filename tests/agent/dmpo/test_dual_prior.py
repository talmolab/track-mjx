"""Scheduled behavior-prior exploration (dual_prior) and the SCAMPER-main checkpoint loader."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics

from track_mjx.agent.dmpo.dual_prior import (
    DualPriorExplore,
    gate_prob,
    init_gate,
    make_behavior_prior_explore_fn,
    step_gate,
)
from track_mjx.agent.dmpo.networks_kl_anchor import _DecoderModule, _PriorModule
from track_mjx.agent.dmpo.rollout import collect_rollout

from .test_rollout import MockEnv, _flat_normalizer, _GaussianDist

_GOLDEN = Path(__file__).parent / "golden" / "scamper_main_s3_parity.npz"
# The SCAMPER-main stage-3 checkpoint the golden was made from (SCAMPER main's own venv,
# analysis/2026-11-04-2-scamper-update-scoping/scripts/make_parity_reference.py). Large; the
# loader test skips when it is not on this machine.
_S3 = Path(
    "/home/talmolab/Desktop/SalkResearch/analysis/2026-11-04-2-scamper-update-scoping/"
    "checkpoints/s3_bp/261105_022735_279614"
)


def _sched(**kw):
    base = dict(
        explore_fn=None,
        p_start=0.0,
        p_end=1.0,
        warmup_env_steps=100.0,
        ramp_env_steps=200.0,
        ramp_shape="cosine",
        bout_tau=25.0,
    )
    base.update(kw)
    return DualPriorExplore(**base)


# ---------------------------------------------------------------- schedule


def test_gate_prob_holds_ramps_and_saturates():
    s = _sched()
    assert float(gate_prob(0.0, s)) == 0.0
    assert float(gate_prob(100.0, s)) == 0.0  # end of warmup
    assert float(gate_prob(200.0, s)) == pytest.approx(0.5)  # cosine midpoint
    assert float(gate_prob(300.0, s)) == pytest.approx(1.0)
    assert float(gate_prob(1e9, s)) == pytest.approx(1.0)
    lin = _sched(ramp_shape="linear", p_start=0.2, p_end=0.6)
    assert float(gate_prob(150.0, lin)) == pytest.approx(0.2 + 0.4 * 0.25)


@pytest.mark.parametrize(
    "bad",
    [
        dict(p_start=-0.1),
        dict(p_end=1.5),
        dict(ramp_env_steps=0.0),
        dict(warmup_env_steps=-1.0),
        dict(ramp_shape="step"),
        dict(bout_tau=0.5),
    ],
)
def test_validate_rejects_bad_schedules(bad):
    with pytest.raises(ValueError):
        _sched(**bad).validate()


def test_gate_chain_is_pinned_at_the_endpoints():
    key = jax.random.key(0)
    g0 = init_gate(key, jnp.float32(0.0), 64)
    g1 = init_gate(key, jnp.float32(1.0), 64)
    assert float(g0.sum()) == 0.0 and float(g1.sum()) == 64.0
    for i in range(50):
        k = jax.random.fold_in(key, i)
        g0 = step_gate(g0, jnp.float32(0.0), 25.0, k)
        g1 = step_gate(g1, jnp.float32(1.0), 25.0, k)
    assert float(g0.sum()) == 0.0 and float(g1.sum()) == 64.0


def test_gate_chain_stationary_fraction_and_bout_length():
    """Stationary P(gate=1) = p; mean bout under the policy = tau / (1 - p)."""
    p, tau, n_envs, n_steps = 0.3, 10.0, 4096, 400
    key = jax.random.key(1)
    g = init_gate(key, jnp.float32(p), n_envs)

    def body(g, k):
        g = step_gate(g, jnp.float32(p), tau, k)
        return g, g

    _, gates = jax.lax.scan(body, g, jax.random.split(key, n_steps))
    gates = np.asarray(gates)
    assert gates.mean() == pytest.approx(p, abs=0.01)
    # Mean length of on-bouts: on-steps / number of off->on and initial-on starts.
    starts = (gates[1:] > 0.5) & (gates[:-1] < 0.5)
    on_runs = starts.sum() + (gates[0] > 0.5).sum()
    assert gates.sum() / on_runs == pytest.approx(tau / (1 - p), rel=0.1)


# ---------------------------------------------------------------- rollout


_EXPLORE_VALUE = 7.0


def _const_explore(policy_params, obs, key):
    del policy_params, obs, key
    return jnp.full((MockEnv.action_size,), _EXPLORE_VALUE)


def _rollout(dual, p, rng=None, num_envs=16, num_steps=12):
    env = MockEnv()
    return collect_rollout(
        env,
        lambda params, obs: _GaussianDist(env.action_size),
        None,
        _flat_normalizer(env.obs_size),
        jax.random.key(3) if rng is None else rng,
        num_envs=num_envs,
        num_steps=num_steps,
        dual_prior=dual,
        dual_prior_p=None if p is None else jnp.float32(p),
    )


def test_rollout_without_dual_prior_returns_three_values():
    assert len(_rollout(None, None)) == 3


def test_p_one_executes_exactly_the_policy_samples():
    base_traj, *_ = _rollout(None, None)
    traj, _, _, stats = _rollout(_sched(explore_fn=_const_explore), 1.0)
    np.testing.assert_array_equal(np.asarray(traj["action"]), base_traj["action"])
    np.testing.assert_array_equal(np.asarray(traj["reward"]), base_traj["reward"])
    assert float(stats["gate_frac"]) == 1.0
    assert "_dual_prior_gate" not in traj


def test_p_zero_executes_and_stores_the_explore_action():
    traj, _, _, stats = _rollout(_sched(explore_fn=_const_explore), 0.0)
    np.testing.assert_array_equal(np.asarray(traj["action"]), _EXPLORE_VALUE)
    assert float(stats["gate_frac"]) == 0.0


def test_intermediate_p_mixes_per_env_in_bouts():
    traj, _, _, stats = _rollout(
        _sched(explore_fn=_const_explore, bout_tau=4.0), 0.5, num_envs=256, num_steps=40
    )
    explored = np.all(np.asarray(traj["action"]) == _EXPLORE_VALUE, axis=-1)  # [N, T]
    assert 0.35 < explored.mean() < 0.65
    assert float(stats["gate_frac"]) == pytest.approx(1.0 - explored.mean(), abs=1e-6)
    # Bouts, not chatter: consecutive steps agree far more often than independent draws (0.5).
    same = (explored[:, 1:] == explored[:, :-1]).mean()
    assert same > 0.7


def test_dual_prior_refuses_behavior_mixing_and_a_missing_p():
    env = MockEnv()
    common = dict(
        env=env,
        policy_apply=lambda params, obs: _GaussianDist(env.action_size),
        policy_params=None,
        normalizer_params=_flat_normalizer(env.obs_size),
        rng=jax.random.key(0),
        num_envs=2,
        num_steps=2,
    )
    with pytest.raises(ValueError, match="dual_prior_p"):
        collect_rollout(**common, dual_prior=_sched(explore_fn=_const_explore))
    with pytest.raises(ValueError, match="behavior mixing"):
        collect_rollout(
            **common,
            dual_prior=_sched(explore_fn=_const_explore),
            dual_prior_p=jnp.float32(0.5),
            frozen_policy_params={},
            behavior_mix_frac=jnp.float32(0.5),
        )


# ---------------------------------------------------------------- explore_fn


def test_explore_fn_samples_the_decoder_at_a_behavior_prior_latent():
    latent, action, proprio_size = 16, 38, 277
    bp_sizes, dec_sizes = (32, 16), (64, 32)
    key = jax.random.key(5)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = jax.random.normal(k1, (proprio_size,))
    bp = _PriorModule(layer_sizes=bp_sizes, latents=latent)
    bp_params = bp.init(k2, x)["params"]
    dec = _DecoderModule(layer_sizes=(*dec_sizes, 2 * action))
    dec_params = dec.init(k3, jnp.zeros(latent + proprio_size))["params"]
    fn = make_behavior_prior_explore_fn(
        behavior_prior_params=bp_params,
        behavior_prior_layer_sizes=bp_sizes,
        decoder_layer_sizes=dec_sizes,
        latent_size=latent,
        action_size=action,
        behavior_logvar=-2.25,
    )
    got = fn({"params": {"decoder": dec_params}}, {"proprioception": x}, k4)

    mu_b, _ = bp.apply({"params": bp_params}, x)
    k_z, k_a = jax.random.split(k4)
    z = mu_b + jnp.exp(0.5 * -2.25) * jax.random.normal(k_z, mu_b.shape)
    out = dec.apply({"params": dec_params}, jnp.concatenate([z, x]))
    want = out[:action] + (jax.nn.softplus(out[action:]) + 1e-3) * jax.random.normal(
        k_a, (action,)
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- loader


@pytest.mark.skipif(
    not _S3.exists(), reason="SCAMPER-main S3 checkpoint not on this machine"
)
def test_scamper_main_loader_matches_scamper_mains_own_inference():
    """Motion prior, behavior prior and decoder logits vs SCAMPER main's own functions."""
    from track_mjx.agent.dmpo.scamper_main_prior import load_scamper_main_checkpoint

    ref = np.load(_GOLDEN)
    sm = load_scamper_main_checkpoint(str(_S3))
    assert sm.step == 47
    nc = sm.config["network_config"]
    xn = running_statistics.normalize(
        jnp.asarray(ref["proprio"]), sm.normalizer_params.proprioception
    )
    lat = int(nc["intention_size"])
    m, lv = _PriorModule(layer_sizes=tuple(nc["prior_layer_sizes"]), latents=lat).apply(
        {"params": sm.motion_prior_params}, xn
    )
    np.testing.assert_allclose(m, ref["mp_mean"], atol=1e-5)
    np.testing.assert_allclose(lv, ref["mp_logvar"], atol=1e-5)
    m, lv = _PriorModule(layer_sizes=tuple(nc["rgm_layer_sizes"]), latents=lat).apply(
        {"params": sm.behavior_prior_params}, xn
    )
    np.testing.assert_allclose(m, ref["bp_mean"], atol=1e-5)
    np.testing.assert_allclose(lv, ref["bp_logvar"], atol=1e-5)
    out = _DecoderModule(
        layer_sizes=(*tuple(nc["decoder_layer_sizes"]), 2 * int(nc["action_size"]))
    ).apply(
        {"params": sm.decoder_params},
        jnp.concatenate([jnp.asarray(ref["latent"]), xn], axis=-1),
    )
    np.testing.assert_allclose(out, ref["logits"], atol=1e-5)
