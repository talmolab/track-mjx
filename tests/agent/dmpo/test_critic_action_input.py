"""Critic action input (``critic_action_input``).

acme wraps its critic in ``ClipToSpec``, so the critic always rates the action
the env executes. Here that action is ``action_utils.bind`` of the pre-tanh
sample. With ``"executed"`` every critic apply -- the critic loss on the stored
action, the bootstrap action(s) and the E-step's sampled actions -- must see a
bound action, while the MPO loss keeps working on the raw samples. A spy
critic records every action it is applied to.
"""

from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.agent.dmpo import test_learner_rnn as rnn_t
from track_mjx.agent.dmpo import learner
from track_mjx.agent.dmpo.action_utils import bind
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    _sgd_step_rnn,
    critic_action,
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.networks import (
    CategoricalCriticHead,
    DMPONetworks,
    make_dmpo_networks,
)

OBS, ACT = 5, 2
# Added to the target policy's mean, so the bootstrap action of the
# mean-action target (the target policy's mode) lies outside [-1, 1] and only
# the bind can bring it inside.
TARGET_MEAN_SHIFT = 2.5


def _shift_target_mean(state, loc_path):
    """``state`` with ``TARGET_MEAN_SHIFT`` added to the target policy's loc
    bias at ``loc_path`` (a tuple of keys under ``params``)."""

    def _shift(params):
        params = jax.tree.map(lambda x: x, params)  # copy the containers
        node = params["params"]
        for key in loc_path:
            node = node[key]
        node["bias"] = node["bias"] + TARGET_MEAN_SHIFT
        return params

    return state._replace(target_policy_params=_shift(state.target_policy_params))


class _Recorder:
    def __init__(self):
        self.seen = []

    def __call__(self, x):
        self.seen.append(np.asarray(x))

    def max_abs(self):
        jax.effects_barrier()
        assert self.seen, "nothing was recorded"
        return max(float(np.max(np.abs(x))) for x in self.seen)

    def clear(self):
        jax.effects_barrier()
        self.seen.clear()


class _SpyCritic(nn.Module):
    """Records the action of every apply. Same parameter layout as the
    recurrent tests' stub critic, so it can rate with its params."""

    num_atoms: int
    vmin: float
    vmax: float
    recorder: object = None
    width: int = 16

    @nn.compact
    def __call__(self, obs, action):
        jax.debug.callback(self.recorder, action)
        if isinstance(obs, Mapping):
            obs = jnp.concatenate([obs["imitation_target"], obs["proprioception"]], -1)
        h = jnp.tanh(nn.Dense(self.width)(jnp.concatenate([obs, action], -1)))
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


def _ff(mode, target_mode, recorder):
    cfg = DMPOConfig(
        policy_layer_sizes=(16,),
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        num_samples=16,
        n_step=2,
        use_n_step=True,
        batch_size=4,
        sequence_length=3,
        store_next_observation=False,
        critic_action_input=mode,
        critic_target_mode=target_mode,
    )
    policy = make_dmpo_networks(OBS, ACT, cfg).policy
    nets = DMPONetworks(
        policy=policy,
        critic=_SpyCritic(num_atoms=11, vmin=-5.0, vmax=5.0, recorder=recorder),
    )
    state = init_training_state(
        jax.random.key(0), nets, {"obs_size": OBS, "action_size": ACT}, cfg
    )
    state = _shift_target_mean(state, ("GaussianPolicyHead_0", "loc"))
    ks = jax.random.split(jax.random.key(1), 3)
    batch = {
        "observation": jax.random.normal(ks[0], (4, 3, OBS)),
        # Stored pre-tanh samples well outside [-1, 1].
        "action": jax.random.uniform(ks[1], (4, 3, ACT), minval=-3.0, maxval=3.0),
        "reward": jax.random.normal(ks[2], (4, 3)),
        "discount": jnp.ones((4, 3)),
        "truncation": jnp.zeros((4, 3)),
    }
    return cfg, nets, state, batch


def test_critic_action_helper():
    a = jnp.array([-3.0, -0.5, 0.0, 2.0])
    raw = DMPOConfig(critic_action_input="raw")
    assert critic_action(a, raw) is a
    np.testing.assert_array_equal(
        critic_action(a, DMPOConfig(critic_action_input="executed")), bind(a)
    )


@pytest.mark.parametrize("target_mode", ["sample_mixture", "mean_action"])
@pytest.mark.parametrize("mode", ["executed", "raw"])
def test_executed_input_applies_bind_at_every_site(mode, target_mode):
    """Feed-forward step: the critic loss, the bootstrap target and the E-step
    Q-values all apply the critic; under "executed" every recorded action is
    inside (-1, 1), under "raw" the stored +-3 actions reach the critic. The
    target policy's mean sits near +2.5, so the mean-action bootstrap action
    is pinned too."""
    rec = _Recorder()
    cfg, nets, state, batch = _ff(mode, target_mode, rec)
    rec.clear()  # drop the init apply
    out = jax.jit(lambda s, b: sgd_step(s, b, nets, make_optimizers(cfg), cfg))(
        state, batch
    )
    jax.block_until_ready(out)
    if mode == "executed":
        assert rec.max_abs() < 1.0
    else:
        assert rec.max_abs() > 1.5


@pytest.mark.parametrize("target_mode", ["sample_mixture", "mean_action"])
@pytest.mark.parametrize("mode", ["executed", "raw"])
def test_executed_input_on_the_recurrent_path(mode, target_mode):
    """As the feed-forward test, through ``_sgd_step_rnn``. The target
    policy's mean is shifted to near +2.5 (the stored hidden comes from the
    online policy, so the batch is unchanged)."""
    rec = _Recorder()
    s = rnn_t._setup(critic_action_input=mode, critic_target_mode=target_mode)
    cfg = s["cfg"]
    state = _shift_target_mean(s["state"], ("loc",))
    nets = DMPONetworks(
        policy=s["nets"].policy,
        critic=_SpyCritic(
            num_atoms=cfg.num_atoms, vmin=cfg.vmin, vmax=cfg.vmax, recorder=rec
        ),
    )
    batch = dict(s["batch"], action=3.0 * s["batch"]["action"] / 0.9)
    out = jax.jit(lambda st, b: _sgd_step_rnn(st, b, nets, s["optimizers"], cfg))(
        state, batch
    )
    jax.block_until_ready(out)
    if mode == "executed":
        assert rec.max_abs() < 1.0
    else:
        assert rec.max_abs() > 1.5


def test_mpo_loss_still_sees_raw_samples(monkeypatch):
    """Under "executed" the MPO loss is still handed the unbounded samples."""
    rec_critic, rec_mpo = _Recorder(), _Recorder()
    cfg, nets, state, batch = _ff("executed", "sample_mixture", rec_critic)
    real_build = learner._build_loss

    class _SpyMPO:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *args, **kwargs):
            actions = kwargs.get("actions", args[3] if len(args) > 3 else None)
            jax.debug.callback(rec_mpo, actions)
            return self.inner(*args, **kwargs)

    monkeypatch.setattr(learner, "_build_loss", lambda c: _SpyMPO(real_build(c)))
    jax.block_until_ready(sgd_step(state, batch, nets, make_optimizers(cfg), cfg))
    assert rec_mpo.max_abs() > 1.0  # 16 x 4 x 2 draws at scale ~0.7
    assert rec_critic.max_abs() < 1.0
