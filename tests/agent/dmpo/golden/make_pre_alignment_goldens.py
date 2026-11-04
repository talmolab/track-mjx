"""Record the DMPO learner/loss outputs of the pre-alignment code as goldens.

The goldens pin the exact numerics of the learner and the MPO loss BEFORE the
acme-alignment switches (``critic_target_mode`` ... ``policy_grad_clip_scope``)
change any behaviour. ``test_legacy_equivalence.py`` recomputes the same
quantities under ``LEGACY_ALIGNMENT`` and compares them bitwise, so a later
change that leaks into the legacy path is caught at once.

Four groups are recorded, each from a fixed seed and tiny nets on CPU:

* ``ff``   -- two feed-forward ``sgd_step`` calls (flat obs, n-step returns,
  one terminal inside the window): every metric and every updated leaf of the
  training state.
* ``rnn``  -- one per-step recurrent ``_sgd_step_rnn`` call (stub GRU policy).
* ``tick`` -- one strided recurrent call (``rnn_tick_stride=2``; stub slow-core
  policy whose hidden carries a phase leaf).
* ``mpo``  -- ``MPO.__call__`` with ``action_penalization=True`` (loss, every
  stat, and the gradients of the loss w.r.t. the dual params and the online
  mean/scale), plus the same call with ``action_penalization=False``.

Regenerating is only meaningful at a commit whose learner still has the
pre-alignment numerics (the parent of the commit that introduced this file).
The script works unchanged on both sides of the alignment fields: when
``LEGACY_ALIGNMENT`` is importable it is applied, otherwise the config has no
such fields and nothing is overridden.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python \
        tests/agent/dmpo/golden/make_pre_alignment_goldens.py
"""

import pathlib

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo import config as _config_module
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    TrainingState,
    _build_loss,
    _sgd_step_rnn,
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.losses import MPO
from track_mjx.agent.dmpo.networks import (
    CategoricalCriticHead,
    DMPONetworks,
    make_dmpo_networks,
)
from track_mjx.agent.dmpo.obs_utils import init_dict_normalizer

tfd = tfp.distributions

GOLDEN_PATH = pathlib.Path(__file__).with_name("pre_alignment.npz")

# Group name -> key prefix inside the .npz.
GROUPS = ("ff", "rnn", "tick", "mpo")

_B, _OBS, _ACT, _HID = 4, 6, 3, 5


def default_overrides() -> dict:
    """The legacy alignment values, or nothing on a pre-alignment checkout."""
    return dict(getattr(_config_module, "LEGACY_ALIGNMENT", {}))


# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------


def _put_tree(out: dict, prefix: str, tree) -> None:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in leaves:
        name = f"{prefix}/{jax.tree_util.keystr(path)}"
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)  # typed PRNG key -> uint32 data
        out[name] = np.asarray(leaf)


def _put_metrics(out: dict, prefix: str, metrics: dict) -> None:
    for k in sorted(metrics):
        out[f"{prefix}/metric/{k}"] = np.asarray(metrics[k])


def _put_state(out: dict, prefix: str, state: TrainingState) -> None:
    for field in TrainingState._fields:
        _put_tree(out, f"{prefix}/state/{field}", getattr(state, field))


def _cfg(overrides: dict, **kw) -> DMPOConfig:
    return DMPOConfig(**kw, **overrides)


# ---------------------------------------------------------------------------
# 1. Feed-forward sgd_step
# ---------------------------------------------------------------------------


def _ff_group(overrides: dict) -> dict:
    cfg = _cfg(
        overrides,
        policy_layer_sizes=(16, 16),
        critic_layer_sizes=(16, 16),
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        num_samples=5,
        n_step=3,
        use_n_step=True,
        discount=0.9,
        batch_size=_B,
        sequence_length=6,
    )
    env_spec = {"obs_size": _OBS, "action_size": _ACT}
    nets = make_dmpo_networks(_OBS, _ACT, cfg)
    state = init_training_state(jax.random.key(0), nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    T = 6
    ks = jax.random.split(jax.random.key(1), 5)
    discount = jnp.ones((_B, T)).at[1, 2].set(0.0)
    batch = {
        "observation": jax.random.normal(ks[0], (_B, T, _OBS)),
        "action": jax.random.uniform(ks[1], (_B, T, _ACT), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[2], (_B, T)),
        "discount": discount,
        "next_observation": jax.random.normal(ks[3], (_B, T, _OBS)),
    }
    step = jax.jit(lambda s, b: sgd_step(s, b, nets, optimizers, cfg))
    out: dict = {}
    for i in (1, 2):
        state, metrics = step(state, batch)
        _put_metrics(out, f"ff/step{i}", metrics)
        _put_state(out, f"ff/step{i}", state)
    return out


# ---------------------------------------------------------------------------
# 2./3. Recurrent sgd_step (per-step and strided). Stub policies honoring the
# learner's only contract with recurrent nets: raw(obs, hidden) ->
# (mu, scale, new_hidden), hidden = tuple of per-leaf arrays.
# ---------------------------------------------------------------------------


class _StubGRU(nn.Module):
    action_size: int
    hidden_size: int = _HID

    def setup(self):
        self.mix = nn.Dense(self.hidden_size, name="mix")
        self.cell = nn.GRUCell(features=self.hidden_size, name="cell")
        self.loc = nn.Dense(self.action_size, name="loc")
        self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_size,))

    def _out(self, y):
        mu = self.loc(y)
        scale = jax.nn.softplus(self.log_std) + 1e-3
        return mu, jnp.broadcast_to(scale, mu.shape)

    def _inp(self, obs):
        return jnp.tanh(
            self.mix(
                jnp.concatenate([obs["imitation_target"], obs["proprioception"]], -1)
            )
        )

    def raw(self, obs, hidden):
        (h,) = hidden
        new_h, y = self.cell(h, self._inp(obs))
        mu, scale = self._out(y)
        return mu, scale, (new_h,)

    def __call__(self, obs, hidden):
        mu, scale, new_hidden = self.raw(obs, hidden)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale), new_hidden


class _StubSlowGRU(_StubGRU):
    """GRU core that updates only where the phase leaf is 0 (a "tick") and
    holds in between; ``tick_stride`` is the clock period."""

    tick_stride: int = 2

    def raw(self, obs, hidden):
        h, phase = hidden
        cand, y = self.cell(h, self._inp(obs))
        is_tick = phase < 0.5  # [1] per env, broadcasts over the H axis
        new_h = jnp.where(is_tick, cand, h)
        new_phase = jnp.mod(phase + 1.0, float(self.tick_stride))
        mu, scale = self._out(new_h)
        return mu, scale, (new_h, new_phase)


class _StubCritic(nn.Module):
    num_atoms: int
    vmin: float
    vmax: float

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate(
            [obs["imitation_target"], obs["proprioception"], action], axis=-1
        )
        h = jnp.tanh(nn.Dense(16)(x))
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


class _SlowMeta:
    """Duck-typed ``recurrent_meta``: the learner reads phase_leaf/tick_stride."""

    def __init__(self, tick_stride: int):
        self.phase_leaf = 1
        self.tick_stride = tick_stride


def _obs_template():
    return {
        "imitation_target": jnp.zeros((4,), jnp.float32),
        "proprioception": jnp.zeros((5,), jnp.float32),
    }


def _rnn_batch(key, T, hidden):
    ks = jax.random.split(key, 4)
    return {
        "observation": {
            "imitation_target": jax.random.normal(ks[0], (_B, T, 4)),
            "proprioception": jax.random.normal(ks[1], (_B, T, 5)),
        },
        "action": jax.random.uniform(ks[2], (_B, T, _ACT), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[3], (_B, T)),
        "discount": jnp.ones((_B, T)).at[1, 1].set(0.0),
        "policy_hidden": hidden,
    }


def _rnn_state(cfg, policy, critic, hidden_dummy, seed):
    k_pol, k_crit = jax.random.split(jax.random.key(seed))
    obs = _obs_template()
    pol_p = policy.init(k_pol, obs, hidden_dummy)
    crit_p = critic.init(k_crit, obs, jnp.zeros((_ACT,), jnp.float32))
    dual = _build_loss(cfg).init_params(_ACT, jnp.float32)
    pol_opt, crit_opt, dual_opt = make_optimizers(cfg)
    return TrainingState(
        policy_params=pol_p,
        critic_params=crit_p,
        target_policy_params=pol_p,
        target_critic_params=crit_p,
        dual_params=dual,
        policy_opt_state=pol_opt.init(pol_p),
        critic_opt_state=crit_opt.init(crit_p),
        dual_opt_state=dual_opt.init(dual),
        normalizer_params=init_dict_normalizer(obs),
        steps=jnp.zeros((), jnp.int32),
        rng=jax.random.key(seed + 100),
    )


def _rnn_group(overrides: dict) -> dict:
    L, n = 3, 4
    cfg = _cfg(
        overrides,
        n_step=n,
        rnn_bptt_length=L,
        sequence_length=L + n,
        use_n_step=True,
        store_next_observation=False,
        num_samples=5,
        discount=0.9,
        vmin=-10.0,
        vmax=10.0,
        num_atoms=11,
        batch_size=_B,
    )
    policy = _StubGRU(action_size=_ACT)
    critic = _StubCritic(num_atoms=cfg.num_atoms, vmin=cfg.vmin, vmax=cfg.vmax)
    nets = DMPONetworks(policy=policy, critic=critic)
    state = _rnn_state(cfg, policy, critic, (jnp.zeros((_HID,)),), seed=2)
    h = 0.3 * jax.random.normal(jax.random.key(3), (_B, L + n, _HID))
    batch = _rnn_batch(jax.random.key(4), L + n, (h,))
    optimizers = make_optimizers(cfg)
    step = jax.jit(lambda s, b: _sgd_step_rnn(s, b, nets, optimizers, cfg))
    out: dict = {}
    new_state, metrics = step(state, batch)
    _put_metrics(out, "rnn/step1", metrics)
    _put_state(out, "rnn/step1", new_state)
    return out


def _tick_group(overrides: dict) -> dict:
    L, n, k = 3, 3, 2
    T = L * k + n
    cfg = _cfg(
        overrides,
        n_step=n,
        rnn_bptt_length=L,
        rnn_tick_stride=k,
        sequence_length=T,
        use_n_step=True,
        store_next_observation=False,
        num_samples=5,
        discount=0.9,
        vmin=-10.0,
        vmax=10.0,
        num_atoms=11,
        batch_size=_B,
    )
    policy = _StubSlowGRU(action_size=_ACT, tick_stride=k)
    critic = _StubCritic(num_atoms=cfg.num_atoms, vmin=cfg.vmin, vmax=cfg.vmax)
    nets = DMPONetworks(policy=policy, critic=critic, recurrent_meta=_SlowMeta(k))
    hidden_dummy = (jnp.zeros((_HID,)), jnp.zeros((1,)))
    state = _rnn_state(cfg, policy, critic, hidden_dummy, seed=5)

    # Stored phases: the clock t mod k from a per-row offset, restarting at 0
    # on the step after a done (what the rollout stores).
    batch = _rnn_batch(jax.random.key(6), T, None)
    done = np.asarray(batch["discount"]) == 0
    phase = np.zeros((_B, T), np.float32)
    start = np.array([0, 1, 0, 1], np.float32)
    for b in range(_B):
        p = start[b]
        for t in range(T):
            phase[b, t] = p
            p = 0.0 if done[b, t] else (p + 1.0) % k
    h = 0.3 * jax.random.normal(jax.random.key(7), (_B, T, _HID))
    batch["policy_hidden"] = (h, jnp.asarray(phase)[..., None])

    optimizers = make_optimizers(cfg)
    step = jax.jit(lambda s, b: _sgd_step_rnn(s, b, nets, optimizers, cfg))
    out: dict = {}
    new_state, metrics = step(state, batch)
    _put_metrics(out, "tick/step1", metrics)
    _put_state(out, "tick/step1", new_state)
    return out


# ---------------------------------------------------------------------------
# 4. MPO.__call__
# ---------------------------------------------------------------------------


def _mpo_group(overrides: dict) -> dict:
    # The penalty formula is the one alignment field MPO itself takes; absent
    # (a pre-alignment checkout) the constructor has no such argument.
    formula = {k: overrides[k] for k in ("action_penalty_formula",) if k in overrides}
    N, B, D = 6, 5, 3
    ks = jax.random.split(jax.random.key(8), 6)
    online_loc = jax.random.normal(ks[0], (B, D))
    online_scale = jax.nn.softplus(jax.random.normal(ks[1], (B, D))) + 0.1
    target_loc = online_loc + 0.1 * jax.random.normal(ks[2], (B, D))
    target_scale = jax.nn.softplus(jax.random.normal(ks[3], (B, D))) + 0.1
    # Out-of-bound actions so the penalty term is not trivially zero.
    actions = 1.5 * jax.random.normal(ks[4], (N, B, D))
    q_values = jax.random.normal(ks[5], (N, B))

    out: dict = {}
    for tag, pen in (("pen", True), ("nopen", False)):
        mpo = MPO(
            epsilon=0.1,
            epsilon_mean=0.0025,
            epsilon_stddev=1e-7,
            epsilon_penalty=0.1,
            init_log_temperature=2.0,
            init_log_alpha_mean=2.0,
            init_log_alpha_stddev=3.0,
            per_dim_constraining=True,
            action_penalization=pen,
            **formula,
        )
        params = mpo.init_params(D, jnp.float32)

        def _run(params, loc, scale, mpo=mpo):
            online = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            target = tfd.MultivariateNormalDiag(loc=target_loc, scale_diag=target_scale)
            loss, stats = mpo(params, online, target, actions, q_values)
            return jnp.sum(loss), (loss, stats)

        (_, (loss, stats)), grads = jax.jit(
            jax.value_and_grad(_run, argnums=(0, 1, 2), has_aux=True)
        )(params, online_loc, online_scale)
        out[f"mpo/{tag}/loss"] = np.asarray(loss)
        for name, val in stats._asdict().items():
            if val is not None:
                out[f"mpo/{tag}/stat/{name}"] = np.asarray(val)
        _put_tree(out, f"mpo/{tag}/grad_params", grads[0])
        out[f"mpo/{tag}/grad_online_loc"] = np.asarray(grads[1])
        out[f"mpo/{tag}/grad_online_scale"] = np.asarray(grads[2])
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BUILDERS = {
    "ff": _ff_group,
    "rnn": _rnn_group,
    "tick": _tick_group,
    "mpo": _mpo_group,
}


def compute_group(group: str, overrides: dict | None = None) -> dict:
    """Recompute one golden group; ``overrides`` are extra DMPOConfig fields."""
    if overrides is None:
        overrides = default_overrides()
    return _BUILDERS[group](overrides)


def compute_all(overrides: dict | None = None) -> dict:
    out: dict = {}
    for g in GROUPS:
        out.update(compute_group(g, overrides))
    return out


def main() -> None:
    if jax.default_backend() != "cpu":
        raise SystemExit("run with JAX_PLATFORMS=cpu: goldens are CPU-bitwise")
    out = compute_all()
    np.savez_compressed(GOLDEN_PATH, **out)
    size = GOLDEN_PATH.stat().st_size
    print(f"wrote {len(out)} arrays to {GOLDEN_PATH} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
