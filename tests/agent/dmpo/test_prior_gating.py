"""Learned behavior/motion-prior gate (prior_gating): hysteresis, the env-side switch, the gated motor policy."""

from pathlib import Path

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _PriorModule,
    make_dmpo_kl_anchor_networks,
)
from track_mjx.agent.dmpo.optim_kl_anchor import label_param_tree
from track_mjx.agent.dmpo.prior_gating import (
    PriorGatingWrapper,
    hysteresis_gate,
    make_dmpo_gated_motor_networks,
)

_S3 = Path(
    "/home/talmolab/Desktop/SalkResearch/analysis/2026-11-04-2-scamper-update-scoping/"
    "checkpoints/s3_bp/261105_022735_279614"
)


# ---------------------------------------------------------------- hysteresis


@pytest.mark.parametrize(
    ("prev", "g", "want"),
    [
        (0.0, 0.0, 0.0),  # off stays off below the upper threshold
        (0.0, 0.69, 0.0),
        (0.0, 0.71, 1.0),  # switches on above bias + margin
        (1.0, 0.5, 1.0),  # on stays on inside the band
        (1.0, 0.31, 1.0),
        (1.0, 0.29, 0.0),  # switches off below bias - margin
        (1.0, -1.0, 0.0),
    ],
)
def test_hysteresis_gate_matches_scampers_bistable_rule(prev, g, want):
    assert float(hysteresis_gate(jnp.float32(prev), jnp.float32(g), 0.5, 0.2)) == want


# ---------------------------------------------------------------- env-side wrapper


@flax.struct.dataclass
class _State:
    obs: dict
    info: dict
    metrics: dict
    reward: jnp.ndarray
    done: jnp.ndarray


class _FakeInner:
    """Records the action it is stepped with; obs mirror the high-level wrappers' flat dict."""

    def __init__(self, action_size, proprio):
        self._a = action_size
        self._p = proprio

    def reset(self, rng):
        return _State(
            obs={"imitation_target": jnp.zeros(3), "proprioception": self._p},
            info={},
            metrics={},
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
        )

    def step(self, state, action):
        info = dict(state.info)
        info["last_action"] = action
        return state.replace(info=info)

    @property
    def action_size(self):
        return self._a

    @property
    def observation_size(self):
        return {"imitation_target": 3, "proprioception": int(self._p.shape[0])}


L, A, P = 4, 6, 5
_W = jax.random.normal(jax.random.key(3), (L + P, 2 * A))


def _prior_fn(p):
    return jnp.tanh(p[:L]), jnp.zeros(L)


def _bp_fn(p):
    return jnp.cos(p[:L]), jnp.zeros(L)


def _dec_logits(zp):
    return zp @ _W, {}


def _wrap(kind, **kw):
    proprio = jax.random.normal(jax.random.key(1), (P,))
    inner = _FakeInner(L if kind == "latent" else A, proprio)
    w = PriorGatingWrapper(
        inner,
        kind=kind,
        prior_fn=_prior_fn,
        bp_fn=_bp_fn,
        decoder_logits_fn=_dec_logits,
        latent_size=L,
        motor_size=A,
        latent_scale=2.0,
        bp_logvar=-200.0,  # deterministic z_BP = mu_BP for exact checks
        **kw,
    )
    return w, proprio


def test_reset_starts_on_the_behavior_prior_and_exposes_the_gate_state():
    w, _ = _wrap("motor")
    st = w.reset(jax.random.key(0))
    assert float(st.info["prev_gate"]) == 0.0
    assert st.obs["imitation_target"].shape == (4,)
    assert float(st.obs["imitation_target"][-1]) == 0.0
    assert w.action_size == A + 1
    assert w.observation_size["imitation_target"] == 4


def test_gate_off_runs_the_behavior_prior_through_the_decoder_motor():
    w, proprio = _wrap("motor")
    st = w.reset(jax.random.key(0))
    act = jnp.concatenate(
        [jnp.full((A,), 0.3), jnp.array([0.0])]
    )  # gate value below the on-threshold
    nxt = w.step(st, act)
    want = jnp.tanh(
        _dec_logits(jnp.concatenate([jnp.cos(proprio[:L]), proprio]))[0][:A]
    )
    np.testing.assert_allclose(
        np.asarray(nxt.info["last_action"]), np.asarray(want), rtol=1e-5, atol=1e-6
    )
    assert float(nxt.info["prev_gate"]) == 0.0
    assert float(nxt.metrics["gate/on"]) == 0.0


def test_gate_on_passes_the_policy_action_through_and_stays_on_in_the_band():
    w, _ = _wrap("motor")
    st = w.reset(jax.random.key(0))
    act = jnp.concatenate([jnp.full((A,), 0.3), jnp.array([0.9])])
    nxt = w.step(st, act)
    np.testing.assert_allclose(np.asarray(nxt.info["last_action"]), 0.3)
    assert float(nxt.info["prev_gate"]) == 1.0
    assert float(nxt.obs["imitation_target"][-1]) == 1.0
    # inside the hysteresis band: stays on
    nxt2 = w.step(nxt, act.at[-1].set(0.4))
    assert float(nxt2.info["prev_gate"]) == 1.0
    np.testing.assert_allclose(np.asarray(nxt2.info["last_action"]), 0.3)
    nxt3 = w.step(nxt2, act.at[-1].set(0.1))
    assert float(nxt3.info["prev_gate"]) == 0.0


def test_latent_kind_gate_off_feeds_the_delta_that_decodes_exactly_z_bp():
    w, proprio = _wrap("latent")
    st = w.reset(jax.random.key(0))
    nxt = w.step(st, jnp.concatenate([jnp.full((L,), 0.5), jnp.array([-0.5])]))
    delta = nxt.info["last_action"]
    z = _prior_fn(proprio)[0] + 2.0 * delta  # what LatentActionWrapper decodes
    np.testing.assert_allclose(
        np.asarray(z), np.asarray(jnp.cos(proprio[:L])), rtol=1e-5, atol=1e-6
    )


def test_behavior_prior_noise_uses_the_fixed_logvar_and_advances_the_key():
    proprio = jax.random.normal(jax.random.key(1), (P,))
    w = PriorGatingWrapper(
        _FakeInner(L, proprio),
        kind="latent",
        prior_fn=_prior_fn,
        bp_fn=_bp_fn,
        decoder_logits_fn=_dec_logits,
        latent_size=L,
        motor_size=A,
        latent_scale=1.0,
        bp_logvar=-2.25,
    )
    st = w.reset(jax.random.key(0))
    act = jnp.concatenate([jnp.zeros(L), jnp.array([0.0])])
    zs, s = [], st
    for _ in range(400):
        s = w.step(s, act)
        zs.append(np.asarray(_prior_fn(proprio)[0] + s.info["last_action"]))
    zs = np.stack(zs)
    np.testing.assert_allclose(zs.std(0), np.exp(0.5 * -2.25), rtol=0.15)
    np.testing.assert_allclose(zs.mean(0), np.asarray(jnp.cos(proprio[:L])), atol=0.06)


def test_gate_obs_off_leaves_the_observation_untouched():
    w, _ = _wrap("motor", gate_obs=False)
    st = w.reset(jax.random.key(0))
    assert st.obs["imitation_target"].shape == (3,)
    assert w.observation_size["imitation_target"] == 3


def test_kind_and_inner_action_size_are_checked():
    with pytest.raises(ValueError, match="kind"):
        _wrap("both")
    proprio = jnp.zeros(P)
    with pytest.raises(ValueError, match="inner action_size"):
        PriorGatingWrapper(
            _FakeInner(A, proprio),
            kind="latent",
            prior_fn=_prior_fn,
            bp_fn=_bp_fn,
            decoder_logits_fn=_dec_logits,
            latent_size=L,
            motor_size=A,
        )


# ---------------------------------------------------------------- gated motor policy


def _net_kwargs():
    return dict(
        latent_size=8,
        vision_shape=(16, 16, 2),
        cfg=DMPOConfig(),
        prior_layer_sizes=(32, 32),
        decoder_layer_sizes=(32, 32),
        policy_head_layer_sizes=(32, 32),
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        residual_mode="sigma_ball",
        residual_scale=3.0,
    )


def test_gated_motor_policy_motor_part_equals_the_ungated_policy_and_gate_starts_at_zero():
    obs = {
        "vision": jax.random.uniform(jax.random.key(4), (2, 16, 16, 2)),
        "imitation_target": jax.random.normal(jax.random.key(5), (2, 13)),
        "proprioception": jax.random.normal(jax.random.key(6), (2, 20)),
    }
    gated = make_dmpo_gated_motor_networks(
        motor_size=18, gate_init_std=0.7, **_net_kwargs()
    )
    plain = make_dmpo_kl_anchor_networks(
        proprio_size=20, task_obs_size=13, action_size=18, **_net_kwargs()
    )
    gp = gated.policy.init(jax.random.key(0), obs)
    pp = {
        "params": {
            k: v
            for k, v in gp["params"].items()
            if k in ("prior", "decoder", "policy_head")
        }
    }
    dg = gated.policy.apply(gp, obs)
    dp = plain.policy.apply(pp, obs)
    assert dg.event_shape == (19,)
    np.testing.assert_allclose(
        np.asarray(dg.mean()[..., :18]), np.asarray(dp.mean()), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(dg.stddev()[..., :18]), np.asarray(dp.stddev()), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(dg.mean()[..., 18]), 0.0, atol=1e-7
    )  # zero-init gate head
    np.testing.assert_allclose(np.asarray(dg.stddev()[..., 18]), 0.7, rtol=1e-5)
    labels = label_param_tree(gp)["params"]
    assert jax.tree.leaves(labels["gate_head"])[0] == "other"
    assert labels["gate_raw_std"] == "other"


# ---------------------------------------------------------------- behavior-prior fn parity


@pytest.mark.skipif(
    not _S3.exists(), reason="SCAMPER-main S3 checkpoint not on this machine"
)
def test_bp_inference_fn_matches_the_dual_prior_behavior_prior_path():
    from brax.training.acme import running_statistics

    from track_mjx.agent.dmpo.prior_gating import make_bp_inference_fn
    from track_mjx.agent.dmpo.scamper_main_prior import load_scamper_main_checkpoint

    sm = load_scamper_main_checkpoint(str(_S3))
    nc = sm.config["network_config"]
    lat = int(nc["intention_size"])
    fn = make_bp_inference_fn(
        sm.behavior_prior_params,
        sm.normalizer_params,
        lat,
        tuple(nc["rgm_layer_sizes"]),
    )
    raw = jax.random.normal(
        jax.random.key(2), (3, int(sm.normalizer_params.proprioception.mean.shape[-1]))
    )
    m, lv = fn(raw)
    xn = running_statistics.normalize(raw, sm.normalizer_params.proprioception)
    m2, lv2 = _PriorModule(layer_sizes=tuple(nc["rgm_layer_sizes"]), latents=lat).apply(
        {"params": sm.behavior_prior_params}, xn
    )
    np.testing.assert_allclose(np.asarray(m), np.asarray(m2), atol=1e-5)
    np.testing.assert_allclose(np.asarray(lv), np.asarray(lv2), atol=1e-5)
