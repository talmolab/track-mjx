"""Tests for the recurrent (GRU) kl-anchor policy networks.

Everything here runs on tiny unbatched shapes — the codebase convention is
`jax.vmap` of an unbatched apply over envs, so unbatched IS the production
code path. The load-bearing test is the step-0 invariant: with spliced
prior+decoder and the zero-init residual head, the recurrent policy must
reproduce the frozen prior->decoder pipeline for ARBITRARY hidden state —
that is what lets an RNN arm start from the same r_anchor = 1.0 point as
every FF arm.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _DecoderModule,
    _PriorModule,
    make_dmpo_kl_anchor_networks,
)
from track_mjx.agent.dmpo.networks_kl_anchor_rnn import (
    RecurrentPolicyMeta,
    make_dmpo_kl_anchor_rnn_networks,
)
from track_mjx.agent.dmpo.optim_kl_anchor import label_param_tree

# Tiny geometry. vision 8x8 is the floor for cnn_channels=(2,4,8) with the
# VisionEncoder default strides (1,1,2): 8 ->conv3x3-> 6 -> 4 ->s2-> 1.
PROPRIO_SIZE = 10
TASK_OBS_SIZE = 6
ACTION_SIZE = 5
LATENT_SIZE = 4
VISION_SHAPE = (8, 8, 2)
PRIOR_LAYERS = (16, 16)
DECODER_LAYERS = (16, 16)
RNN_MLP_LAYERS = (16,)
RNN_HIDDEN_SIZES = (8, 6)


@pytest.fixture
def cfg():
    return DMPOConfig(
        num_envs=4,
        unroll_length=4,
        batch_size=4,
        sequence_length=4,
    )


def _make_rnn_nets(cfg, **overrides):
    kwargs = dict(
        proprio_size=PROPRIO_SIZE,
        task_obs_size=TASK_OBS_SIZE,
        action_size=ACTION_SIZE,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=PRIOR_LAYERS,
        decoder_layer_sizes=DECODER_LAYERS,
        rnn_cell="gru",
        rnn_mlp_layers=RNN_MLP_LAYERS,
        rnn_hidden_sizes=RNN_HIDDEN_SIZES,
        rnn_store_dtype="float16",
        cnn_feature_size=4,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
    )
    kwargs.update(overrides)
    return make_dmpo_kl_anchor_rnn_networks(**kwargs)


def _random_obs(key, batch=None):
    shape = () if batch is None else (batch,)
    k_v, k_t, k_p = jax.random.split(key, 3)
    return {
        "vision": jax.random.uniform(k_v, shape + VISION_SHAPE),
        "imitation_target": jax.random.normal(k_t, shape + (TASK_OBS_SIZE,)),
        "proprioception": jax.random.normal(k_p, shape + (PROPRIO_SIZE,)),
    }


def test_init_apply_shapes_and_hidden_evolution(cfg):
    nets = _make_rnn_nets(cfg)
    meta = nets.recurrent_meta
    assert isinstance(meta, RecurrentPolicyMeta)
    assert meta.cell_type == "gru"
    assert meta.hidden_sizes == RNN_HIDDEN_SIZES
    assert meta.store_dtype == jnp.dtype("float16")

    # init_hidden per the pinned contract: unbatched [H_l] / batched [B, H_l].
    hidden = meta.init_hidden()
    assert isinstance(hidden, tuple)
    assert [h.shape for h in hidden] == [(8,), (6,)]
    hidden_b = meta.init_hidden(3)
    assert [h.shape for h in hidden_b] == [(3, 8), (3, 6)]

    obs = _random_obs(jax.random.PRNGKey(1))
    params = nets.policy.init(jax.random.PRNGKey(0), obs, hidden)
    dist, new_hidden = nets.policy.apply(params, obs, hidden)

    sample = dist.sample(seed=jax.random.PRNGKey(2))
    assert sample.shape == (ACTION_SIZE,)
    assert isinstance(new_hidden, tuple)
    assert [h.shape for h in new_hidden] == [(8,), (6,)]

    # The GRU must actually move: nonzero update from zero hidden, and a
    # different obs must produce a different hidden (the memory channel is
    # obs-driven even while the zero-init residual output is still 0).
    obs2 = _random_obs(jax.random.PRNGKey(3))
    _, new_hidden2 = nets.policy.apply(params, obs2, hidden)
    for h_new, h_old in zip(new_hidden, hidden):
        assert not np.allclose(np.asarray(h_new), np.asarray(h_old))
    for h_a, h_b in zip(new_hidden, new_hidden2):
        assert not np.allclose(np.asarray(h_a), np.asarray(h_b))


def test_raw_is_vmappable_over_envs(cfg):
    """Per-env vmap of the unbatched apply — the rollout/learner convention."""
    nets = _make_rnn_nets(cfg)
    meta = nets.recurrent_meta
    obs = _random_obs(jax.random.PRNGKey(1))
    params = nets.policy.init(jax.random.PRNGKey(0), obs, meta.init_hidden())

    batch = 3
    obs_b = _random_obs(jax.random.PRNGKey(4), batch=batch)
    hidden_b = meta.init_hidden(batch)
    mu, scale, new_hidden = jax.vmap(
        lambda o, h: nets.policy.apply(params, o, h, method="raw")
    )(obs_b, hidden_b)
    assert mu.shape == (batch, ACTION_SIZE)
    assert scale.shape == (batch, ACTION_SIZE)
    assert [h.shape for h in new_hidden] == [(batch, 8), (batch, 6)]


def test_param_tree_blocks_and_labels(cfg):
    nets = _make_rnn_nets(cfg)
    obs = _random_obs(jax.random.PRNGKey(1))
    params = nets.policy.init(
        jax.random.PRNGKey(0), obs, nets.recurrent_meta.init_hidden()
    )

    # The frozen-block machinery keys off exactly these top-level names.
    assert set(params["params"].keys()) == {"prior", "decoder", "policy_head"}

    labels = label_param_tree(params)
    head_labels = jax.tree_util.tree_leaves(labels["params"]["policy_head"])
    assert len(head_labels) > 0
    assert all(lbl == "policy_head" for lbl in head_labels)


def test_warm_start_splice_grafts_prior_and_decoder(cfg):
    # Donor: the FF kl-anchor net with the same prior/decoder geometry.
    ff_nets = make_dmpo_kl_anchor_networks(
        proprio_size=PROPRIO_SIZE,
        task_obs_size=TASK_OBS_SIZE,
        action_size=ACTION_SIZE,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=PRIOR_LAYERS,
        decoder_layer_sizes=DECODER_LAYERS,
        policy_head_layer_sizes=(16,),
        cnn_feature_size=4,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
    )
    obs = _random_obs(jax.random.PRNGKey(1))
    ff_params = ff_nets.policy.init(jax.random.PRNGKey(7), obs)
    prior_params = ff_params["params"]["prior"]
    decoder_params = ff_params["params"]["decoder"]

    nets = _make_rnn_nets(
        cfg,
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
    )
    # A DIFFERENT init key: the spliced subtrees must come from the donor,
    # not happen to coincide by seeding.
    params = nets.policy.init(
        jax.random.PRNGKey(11), obs, nets.recurrent_meta.init_hidden()
    )

    jax.tree_util.tree_map(
        np.testing.assert_array_equal, params["params"]["prior"], prior_params
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, params["params"]["decoder"], decoder_params
    )


def test_step0_invariant_for_arbitrary_hidden(cfg):
    """Spliced prior+decoder + fresh zero-init residual head => the policy
    dist equals the frozen prior->decoder pipeline for ANY hidden state.

    This is the property the whole design leans on: r_anchor = 1.0000 at
    startup does not depend on the hidden being zeros, so hidden can be
    zero-init'd, resumed, or garbage without breaking the anchor probe.
    """
    donor = _make_rnn_nets(cfg)
    obs = _random_obs(jax.random.PRNGKey(1))
    donor_params = donor.policy.init(
        jax.random.PRNGKey(5), obs, donor.recurrent_meta.init_hidden()
    )
    prior_params = donor_params["params"]["prior"]
    decoder_params = donor_params["params"]["decoder"]

    nets = _make_rnn_nets(
        cfg,
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
    )
    params = nets.policy.init(
        jax.random.PRNGKey(9), obs, nets.recurrent_meta.init_hidden()
    )

    # Frozen pipeline computed directly: prior(proprio) -> decoder([z, proprio]),
    # residual identically 0.
    proprio = obs["proprioception"]
    z_prior, _ = _PriorModule(layer_sizes=PRIOR_LAYERS, latents=LATENT_SIZE).apply(
        {"params": prior_params}, proprio
    )
    decoder_out = _DecoderModule(
        layer_sizes=DECODER_LAYERS + (2 * ACTION_SIZE,)
    ).apply({"params": decoder_params}, jnp.concatenate([z_prior, proprio], axis=-1))
    mu_ref = decoder_out[..., :ACTION_SIZE]
    scale_ref = jax.nn.softplus(decoder_out[..., ACTION_SIZE:]) + 1e-3

    key = jax.random.PRNGKey(21)
    for i in range(3):  # zeros AND two arbitrary random hiddens
        if i == 0:
            hidden = nets.recurrent_meta.init_hidden()
        else:
            key, k = jax.random.split(key)
            hidden = tuple(
                jax.random.normal(kk, (h,)) * 3.0
                for kk, h in zip(jax.random.split(k, len(RNN_HIDDEN_SIZES)),
                                 RNN_HIDDEN_SIZES)
            )
        dist, _ = nets.policy.apply(params, obs, hidden)
        np.testing.assert_allclose(
            np.asarray(dist.mean()), np.asarray(mu_ref), atol=1e-6,
            err_msg=f"step-0 mean invariant violated for hidden #{i}",
        )
        np.testing.assert_allclose(
            np.asarray(dist.stddev()), np.asarray(scale_ref), atol=1e-6,
            err_msg=f"step-0 stddev invariant violated for hidden #{i}",
        )


def test_raw_method_consistent_with_call(cfg):
    nets = _make_rnn_nets(cfg)
    meta = nets.recurrent_meta
    obs = _random_obs(jax.random.PRNGKey(1))
    params = nets.policy.init(jax.random.PRNGKey(0), obs, meta.init_hidden())

    key = jax.random.PRNGKey(6)
    hidden = tuple(
        jax.random.normal(kk, (h,))
        for kk, h in zip(jax.random.split(key, len(RNN_HIDDEN_SIZES)),
                         RNN_HIDDEN_SIZES)
    )
    mu, scale, nh_raw = nets.policy.apply(params, obs, hidden, method="raw")
    dist, nh_call = nets.policy.apply(params, obs, hidden)

    np.testing.assert_allclose(np.asarray(dist.mean()), np.asarray(mu), atol=1e-6)
    np.testing.assert_allclose(np.asarray(dist.stddev()), np.asarray(scale), atol=1e-6)
    for h_r, h_c in zip(nh_raw, nh_call):
        np.testing.assert_allclose(np.asarray(h_r), np.asarray(h_c), atol=1e-6)


def test_non_gru_cell_fails_loud(cfg):
    with pytest.raises(ValueError, match="gru"):
        _make_rnn_nets(cfg, rnn_cell="lstm")
