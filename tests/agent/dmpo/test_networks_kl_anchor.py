import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_kl_anchor import (
    make_dmpo_kl_anchor_networks,
)


@pytest.fixture
def cfg():
    return DMPOConfig(
        num_envs=4,
        unroll_length=4,
        batch_size=4,
        sequence_length=4,
    )


def test_warm_start_invariant_at_step_zero(cfg):
    """At t=0 with warm-started prior+decoder weights and zero residual, the
    trainable pipeline's PRE-TANH μ_θ must equal the imit decoder's logits-mean
    on imit-normalized proprio, AND post-tanh action must equal NormalTanhDistribution.mode.

    This test simulates production conditions: the trainable branch receives
    proprio normalized by the imit normalizer (because that's what
    `seed_proprio_from_imit` makes happen at runtime), and we compare against
    the actual brax `NormalTanhDistribution.mode(...)` — the ground-truth
    semantics of the imit decoder's output.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.training import distribution
    from brax.training.acme import running_statistics
    from track_mjx.agent.dmpo.action_utils import bind
    from track_mjx.agent.dmpo.networks_kl_anchor import (
        make_dmpo_kl_anchor_networks,
    )
    from scamper.agent.mlp_prior.prior_networks import Prior as ScamperPrior
    from scamper.agent.imitation.intention_network import Decoder as ScamperDecoder

    rng = jax.random.PRNGKey(7)
    proprio_size = 32
    task_obs_size = 12
    action_size = 18
    vision_shape = (16, 16, 2)
    latent_size = 8
    prior_layers = (32, 32)
    decoder_layers = (32, 32)

    # 1. Build a *real* imit-style prior + decoder; init with random params.
    frozen_prior = ScamperPrior(layer_sizes=list(prior_layers), latents=latent_size)
    frozen_decoder = ScamperDecoder(
        layer_sizes=list(decoder_layers) + [2 * action_size],
    )
    rng, k_p, k_d = jax.random.split(rng, 3)
    prior_init = frozen_prior.init(k_p, jnp.zeros((1, proprio_size)))
    decoder_init = frozen_decoder.init(
        k_d, jnp.zeros((1, latent_size + proprio_size))
    )
    prior_params = prior_init["params"]
    decoder_params = decoder_init["params"]

    # 2. Build a non-trivial imit-style proprio normalizer.
    rng, k_n = jax.random.split(rng)
    proprio_mean = jax.random.normal(k_n, (proprio_size,)) * 0.5
    proprio_std = 0.5 + jnp.abs(jax.random.normal(k_n, (proprio_size,))) * 0.5
    proprio_norm = running_statistics.RunningStatisticsState(
        mean=proprio_mean,
        std=proprio_std,
        count=jnp.array(1_000_000.0, dtype=jnp.float32),
        summed_variance=proprio_std * proprio_std * 1_000_000.0,
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )

    # 3. Sample non-zero proprio (in raw env units, NOT normalized).
    rng, k_obs = jax.random.split(rng)
    raw_proprio = jax.random.normal(k_obs, (1, proprio_size))

    # 4. Compute the IMIT mode action: prior(normalize(p)) → decoder([z, normalize(p)])
    #    → NormalTanhDistribution.mode(logits) = tanh(logits[..., :action_size]).
    norm_proprio = running_statistics.normalize(raw_proprio, proprio_norm)
    z_imit, _ = frozen_prior.apply({"params": prior_params}, norm_proprio)
    decoder_input = jnp.concatenate([z_imit, norm_proprio], axis=-1)
    frozen_logits, _ = frozen_decoder.apply({"params": decoder_params}, decoder_input)
    parametric = distribution.NormalTanhDistribution(event_size=action_size)
    a_imit = parametric.mode(frozen_logits)  # tanh(μ_pretanh)

    # 5. Build the trainable kl-anchor pipeline with the SAME weights spliced in.
    nets = make_dmpo_kl_anchor_networks(
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=prior_layers,
        decoder_layer_sizes=decoder_layers,
        policy_head_layer_sizes=(32, 32),
        cfg=cfg,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
    )
    # Initialise the trainable policy. The policy_head's last Dense is
    # zero-init'd so the residual is 0 at step 0 — that's the warm-start
    # invariant we're testing.
    obs_normalized = {
        "vision": jnp.zeros((1,) + vision_shape),
        "imitation_target": jnp.zeros((1, task_obs_size)),
        # The trainable pipeline expects normalized proprio — the runtime
        # delivers this via the seeded DMPO normalizer. We pass it directly.
        "proprioception": norm_proprio,
    }
    params = nets.policy.init(jax.random.PRNGKey(0), obs_normalized)
    dist = nets.policy.apply(params, obs_normalized)
    mu_theta = dist.loc          # raw (pre-tanh) mean
    a_theta = bind(mu_theta)     # post-tanh, matching what env sees

    # 6. Assertion: post-tanh trainable action equals imit mode action.
    np.testing.assert_allclose(
        np.asarray(a_theta), np.asarray(a_imit), atol=1e-4,
        err_msg="Warm-start invariant violated: bind(μ_θ) != NormalTanhDist.mode(imit_logits)",
    )

    # 7. Also assert the scale matches the NormalTanhDistribution interpretation
    #    (softplus + min_std), proving the σ-parameterization fix is wired right.
    expected_scale = jax.nn.softplus(frozen_logits[..., action_size:]) + 1e-3
    # tfp's MultivariateNormalDiag does not expose `scale_diag` as a public
    # attribute on this tfp version; `stddev()` returns the same tensor
    # bit-for-bit (for a diagonal Gaussian, stddev == scale_diag).
    np.testing.assert_allclose(
        np.asarray(dist.stddev()), np.asarray(expected_scale), atol=1e-5,
        err_msg="σ_θ != softplus(log_σ_imit) + 1e-3",
    )


def test_critic_init_runs_without_error(cfg):
    proprio_size = 16
    task_obs_size = 12
    action_size = 18
    vision_shape = (16, 16, 2)

    nets = make_dmpo_kl_anchor_networks(
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
        latent_size=8,
        vision_shape=vision_shape,
        prior_layer_sizes=(32, 32),
        decoder_layer_sizes=(32, 32),
        policy_head_layer_sizes=(32, 32),
        cfg=cfg,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        warm_start_prior_params=None,
        warm_start_decoder_params=None,
    )

    obs = {
        "vision": jnp.zeros((1,) + vision_shape),
        "imitation_target": jnp.zeros((1, task_obs_size)),
        "proprioception": jnp.zeros((1, proprio_size)),
    }
    action = jnp.zeros((1, action_size))
    params = nets.critic.init(jax.random.PRNGKey(0), obs, action)
    dist = nets.critic.apply(params, obs, action)
    assert dist.logits.shape == (1, cfg.num_atoms)


def test_scale_parameterization_matches_normaltanh(cfg):
    """The trainable policy must use softplus(log_std) + 1e-3, matching the
    brax NormalTanhDistribution that the imit decoder was trained against.

    With log_std raw = 0 across all action dims, the policy's scale must be
    softplus(0) + 1e-3 ≈ 0.6941, NOT exp(0) = 1.0.
    """
    proprio_size = 16
    task_obs_size = 12
    action_size = 8
    vision_shape = (16, 16, 2)
    latent_size = 8

    nets = make_dmpo_kl_anchor_networks(
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=(32, 32),
        decoder_layer_sizes=(32, 32),
        policy_head_layer_sizes=(32, 32),
        cfg=cfg,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        warm_start_prior_params=None,
        warm_start_decoder_params=None,
    )

    obs = {
        "vision": jnp.zeros((1,) + vision_shape),
        "imitation_target": jnp.zeros((1, task_obs_size)),
        "proprioception": jnp.zeros((1, proprio_size)),
    }
    params = nets.policy.init(jax.random.PRNGKey(0), obs)

    # Override decoder's last-layer bias to make log_std raw = 0 deterministically.
    inner = dict(params["params"])
    decoder_inner = dict(inner["decoder"])
    last_layer_name = f"hidden_{2}"  # decoder_layer_sizes=(32,32)+output → last is hidden_2
    last_layer = dict(decoder_inner[last_layer_name])
    last_layer["bias"] = jnp.zeros_like(last_layer["bias"])
    last_layer["kernel"] = jnp.zeros_like(last_layer["kernel"])
    decoder_inner[last_layer_name] = last_layer
    inner["decoder"] = decoder_inner
    params = {"params": inner}

    dist = nets.policy.apply(params, obs)
    expected_scale = float(jax.nn.softplus(jnp.float32(0.0))) + 1e-3
    # tfp's MultivariateNormalDiag does not expose `scale_diag` as a public
    # attribute on this tfp version; `stddev()` returns the same tensor
    # bit-for-bit (for a diagonal Gaussian, stddev == scale_diag).
    actual_scale = float(jnp.mean(dist.stddev()))
    # exp(0) = 1.0 vs softplus(0)+1e-3 ≈ 0.6941 — large enough gap that this
    # assertion fails clearly with the buggy `exp` code.
    assert abs(actual_scale - expected_scale) < 1e-4, (
        f"scale should be softplus(0)+1e-3 ≈ {expected_scale:.4f}, got {actual_scale:.4f}"
    )
