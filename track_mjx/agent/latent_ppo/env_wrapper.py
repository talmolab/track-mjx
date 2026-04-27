"""Wrap a Brax/mjx env so each step produces z_sim, z_target, r_mimic.

The wrapper is duck-typed against the existing rat imitation env interface:
  state = env.reset(rng)
  state = env.step(state, action)
where state has .data (mjx.Data with .qpos, .qvel), .obs (dict), .reward,
.done, .info, .metrics.

Outputs (post-wrap):
  state.obs is a flat dict with keys 'proprioception' (flat), 'o_history'
  (flat), 'z_target' (latent_dim,). The original task_obs and the deep
  proprioception OrderedDict are dropped — the policy ONLY consumes the
  three flat keys above (matches LatentMimicPolicy._flatten_obs).
  state.reward = r_mimic.
  state.info['latent_buf'] = _Buffers (motion_window, history, prev_z_mean,
  prev_z_logvar). state.info['mimic_kl'] and state.info['r_mimic'] are
  scalars for logging.
"""
from dataclasses import dataclass
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import struct

from track_mjx.agent.latent_ppo.prior_module import FrozenLatentPrior


def _frame_from_qpos_qvel(qpos: jnp.ndarray, qvel: jnp.ndarray,
                          use_qvel: bool) -> jnp.ndarray:
    """Build a single per-frame motion descriptor from (qpos, qvel).

    use_qvel=True  -> m = (p, theta, v, q, qdot)            feat = 3+4+6+nj+nj
    use_qvel=False -> m = (p, theta, q)                     feat = 3+4+nj
    """
    p = qpos[..., :3]
    theta = qpos[..., 3:7]
    q = qpos[..., 7:]
    if not use_qvel:
        return jnp.concatenate([p, theta, q], axis=-1)
    v = qvel[..., :6]
    qdot = qvel[..., 6:]
    return jnp.concatenate([p, theta, v, q, qdot], axis=-1)


def _kl_diag_gauss(mu_a, lv_a, mu_b, lv_b):
    """Sum-over-dims KL(N(mu_a, exp(lv_a)) || N(mu_b, exp(lv_b)))."""
    var_a = jnp.exp(lv_a)
    var_b = jnp.exp(lv_b)
    return jnp.sum(0.5 * (
        lv_b - lv_a + (var_a + (mu_a - mu_b) ** 2) / var_b - 1.0
    ), axis=-1)


def _mean_kl(mu_a, mu_b):
    """KL between two unit-variance Gaussians = 0.5 * ||mu_a - mu_b||^2.

    This is what we use for r_mimic when the encoder has artificially-tight
    posteriors (v8: sigma_max=0.05 → var=0.0025 → full KL blows up by 400×
    even for tiny mean differences, saturating r_mimic to ~0).
    """
    return 0.5 * jnp.sum((mu_a - mu_b) ** 2, axis=-1)


def _flatten_proprioception(prop) -> jnp.ndarray:
    """Take an arbitrarily-nested mapping/array tree and ravel to (N,)."""
    flat, _ = jax.flatten_util.ravel_pytree(prop)
    return flat


@struct.dataclass
class _Buffers:
    motion_window: jnp.ndarray   # (w, feat_dim) — RAW frames, NOT normalized
    history: jnp.ndarray          # (H, history_dim)
    prev_z_mean: jnp.ndarray      # (latent_dim,)
    prev_z_logvar: jnp.ndarray    # (latent_dim,)


class LatentMimicEnvWrapper:
    """Wrap a base imitation env to expose r_mimic + z_target.

    The wrapper holds the FrozenLatentPrior on the host side (encoder is jit-
    compiled when called inside reset/step). All per-env state (windows,
    history) lives in state.info under the key 'latent_buf' so JAX vmap
    treats it correctly.
    """

    def __init__(
        self,
        env,
        prior_dir: str,
        n_joints: int,
        w_r: float,
        history_len: int,
        prepare_observation_size: bool = True,
        kl_mode: str = "mean",   # 'mean' = unit-var KL = 0.5 ||mu_t - mu_s||^2
                                 # 'full' = full Gaussian KL (paper-faithful but
                                 # explodes when prior has tight posteriors)
    ):
        self.env = env
        self.prior = FrozenLatentPrior.from_dir(prior_dir)
        self.n_joints = n_joints
        self.w_r = float(w_r)
        self.H = int(history_len)
        self.kl_mode = str(kl_mode)
        if self.kl_mode not in ("mean", "full"):
            raise ValueError(f"unknown kl_mode {kl_mode!r}; expected 'mean' or 'full'")

        if self.prior.n_joints != n_joints:
            raise ValueError(
                f"prior n_joints {self.prior.n_joints} != env n_joints {n_joints}"
            )

        # History feature per frame: (q [nj], qdot [nj], prev_action [action_size]).
        # action_size != n_joints in general (rat: 67 qpos joints, 32 actuators).
        self.action_dim = int(env.action_size)
        self.history_dim = 2 * n_joints + self.action_dim

        # Pre-compute proprioception flat dim by running a dry reset; the Brax
        # PPO trainer queries observation_size BEFORE the first env.reset(), so
        # caching it lazily on first reset is too late for network init.
        if prepare_observation_size:
            try:
                _probe = self.env.reset(jax.random.PRNGKey(0))
                _prop = _probe.obs["state"]["proprioception"]
                _flat, _ = jax.flatten_util.ravel_pytree(_prop)
                self._prop_dim_cached = int(_flat.shape[-1])
            except Exception as e:  # noqa: BLE001 — keep wrapper usable even if probe fails
                self._prop_dim_cached = None
                import logging as _logging
                _logging.warning(f"LatentMimicEnvWrapper proprioception probe failed: {e}")

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def unwrapped(self):
        # Critical: do NOT delegate to self.env.unwrapped. Brax's
        # BraxDomainRandomizationVmapWrapper inspects `self.env.unwrapped` and
        # calls reset on it, bypassing every wrapper above. We need our reset
        # in the chain, so we declare ourselves as the unwrapped env. This is
        # safe because the only thing the domain-randomization wrapper does
        # with `unwrapped` is set/restore `_mjx_model`, which we forward via
        # __getattr__ to the inner env.
        return self

    @property
    def observation_size(self):
        # Conform to track-mjx's existing two-key schema (task_obs, proprioception)
        # so ff_ppo.observation_utils.flatten_obs_dict / get_obs_sizes work
        # unchanged. We pack o_history INTO proprioception as a sub-key so the
        # full state is one flat vector after flatten_obs_dict.
        return {
            "state": {
                "task_obs": self.prior.latent_dim,           # z_target
                "proprioception": (
                    self._prop_dim_cached + self.H * self.history_dim
                    if self._prop_dim_cached is not None
                    else None
                ),
            }
        }

    # ------------------------- buffer helpers -------------------------

    def _empty_buffers(self) -> _Buffers:
        return _Buffers(
            motion_window=jnp.zeros((self.prior.window_len, self.prior.feat_dim)),
            history=jnp.zeros((self.H, self.history_dim)),
            prev_z_mean=jnp.zeros((self.prior.latent_dim,)),
            prev_z_logvar=jnp.zeros((self.prior.latent_dim,)),
        )

    def _push_frame(self, buf: _Buffers, qpos, qvel, prev_action) -> _Buffers:
        m = _frame_from_qpos_qvel(qpos, qvel, self.prior.use_qvel)
        new_motion = jnp.concatenate(
            [buf.motion_window[1:], m[None, :]], axis=0
        )
        # history: (q, qdot, prev_action)
        hist_vec = jnp.concatenate(
            [qpos[7:], qvel[6:], prev_action], axis=-1
        )
        new_hist = jnp.concatenate(
            [buf.history[1:], hist_vec[None, :]], axis=0
        )
        return buf.replace(motion_window=new_motion, history=new_hist)

    # ------------------------- prior calls ----------------------------

    def _compute_z_sim(self, motion_window: jnp.ndarray):
        """motion_window: (w, feat_dim) RAW. Returns (mean, logvar) shape (latent_dim,)."""
        mean, logvar = self.prior.encode(motion_window[None, ...])  # adds batch
        return mean[0], logvar[0]

    def _compute_z_target(self, prev_z_mean: jnp.ndarray):
        """prev_z_mean: (latent_dim,). Returns (mean, logvar) of z_target.

        z_target = E(predictor(prev_z_mean)). Predictor output is in NORMALIZED
        space, so we use encode_normalized() which skips the normalizer.
        """
        pred_window = self.prior.predict(prev_z_mean[None, :])  # (1, horizon, feat)
        # The predictor outputs `horizon` frames; the encoder consumes a
        # `window_len`-frame window. If horizon != window_len we cannot
        # directly re-encode (paper recipe assumes they match). For our v8:
        # window_len=10, horizon=5. Pad by repeating the last predicted
        # frame to length window_len.
        horizon = pred_window.shape[1]
        if horizon < self.prior.window_len:
            pad = jnp.broadcast_to(
                pred_window[:, -1:, :],
                (1, self.prior.window_len - horizon, pred_window.shape[-1]),
            )
            pred_window = jnp.concatenate([pred_window, pad], axis=1)
        elif horizon > self.prior.window_len:
            pred_window = pred_window[:, : self.prior.window_len]
        mean, logvar = self.prior.encode_normalized(pred_window)
        return mean[0], logvar[0]

    # ------------------------- reset / step ---------------------------

    def reset(self, rng):
        state = self.env.reset(rng)
        qpos = state.data.qpos
        qvel = state.data.qvel
        prev_action = jnp.zeros((self.action_dim,))

        buf = self._empty_buffers()
        # Fill the rolling window with the initial pose so the encoder sees a
        # sensible (non-zero) window from the first step.
        for _ in range(self.prior.window_len):
            buf = self._push_frame(buf, qpos, qvel, prev_action)
        for _ in range(self.H):
            buf = self._push_frame(buf, qpos, qvel, prev_action)

        z_sim_mean, z_sim_logvar = self._compute_z_sim(buf.motion_window)
        buf = buf.replace(prev_z_mean=z_sim_mean, prev_z_logvar=z_sim_logvar)
        z_t_mean, z_t_logvar = self._compute_z_target(buf.prev_z_mean)

        # Build new obs in the {state: {task_obs, proprioception}} schema that
        # ff_ppo's observation_utils expects.  task_obs = z_target.
        # proprioception = concat(base_proprioception_flat, o_history_flat).
        prop_flat = _flatten_proprioception(state.obs["state"]["proprioception"])
        if not hasattr(self, "_prop_dim_cached") or self._prop_dim_cached is None:
            self._prop_dim_cached = int(prop_flat.shape[-1])
        full_proprio = jnp.concatenate(
            [prop_flat, buf.history.reshape(-1)], axis=-1
        )

        from collections import OrderedDict
        new_obs = OrderedDict(
            state=OrderedDict(
                task_obs=z_t_mean,
                proprioception=full_proprio,
            )
        )

        info = dict(state.info) if state.info else {}
        info["latent_buf"] = buf
        info["r_mimic"] = jnp.float32(1.0)
        info["mimic_kl"] = jnp.float32(0.0)

        metrics = dict(state.metrics) if state.metrics else {}
        metrics["r_mimic"] = jnp.float32(1.0)
        metrics["mimic_kl"] = jnp.float32(0.0)

        return state.replace(
            obs=new_obs,
            reward=jnp.float32(1.0),
            info=info,
            metrics=metrics,
        )

    def step(self, state, action):
        new_state = self.env.step(state, action)
        buf: _Buffers = state.info["latent_buf"]
        qpos = new_state.data.qpos
        qvel = new_state.data.qvel
        buf = self._push_frame(buf, qpos, qvel, action)

        z_sim_mean, z_sim_logvar = self._compute_z_sim(buf.motion_window)
        z_t_mean, z_t_logvar = self._compute_z_target(buf.prev_z_mean)
        if self.kl_mode == "full":
            kl = _kl_diag_gauss(z_t_mean, z_t_logvar, z_sim_mean, z_sim_logvar)
        else:
            kl = _mean_kl(z_t_mean, z_sim_mean)
        r_mimic = jnp.exp(-self.w_r * kl)

        buf = buf.replace(prev_z_mean=z_sim_mean, prev_z_logvar=z_sim_logvar)

        prop_flat = _flatten_proprioception(new_state.obs["state"]["proprioception"])
        full_proprio = jnp.concatenate(
            [prop_flat, buf.history.reshape(-1)], axis=-1
        )

        from collections import OrderedDict
        new_obs = OrderedDict(
            state=OrderedDict(
                task_obs=z_t_mean,
                proprioception=full_proprio,
            )
        )

        info = dict(new_state.info) if new_state.info else {}
        info["latent_buf"] = buf
        info["r_mimic"] = r_mimic
        info["mimic_kl"] = kl

        # Promote r_mimic / mimic_kl into metrics so brax PPO's eval evaluator
        # picks them up as eval/episode_rewards/r_mimic etc., visible in wandb
        # at every eval (not just at the policy_params_fn callback every 10M).
        metrics = dict(new_state.metrics) if new_state.metrics else {}
        metrics["r_mimic"] = r_mimic
        metrics["mimic_kl"] = kl

        return new_state.replace(
            obs=new_obs,
            reward=r_mimic,
            info=info,
            metrics=metrics,
        )

    # Pass-through for the rest (render etc.)
    def __getattr__(self, name):
        return getattr(self.env, name)
