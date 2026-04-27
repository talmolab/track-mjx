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
    """Sum-over-dims KL(N(mu_a, exp(lv_a)) || N(mu_b, exp(lv_b))).

    Per-batch scalar. lv_a and lv_b must be broadcast-compatible with mu_*.
    """
    var_a = jnp.exp(lv_a)
    var_b = jnp.exp(lv_b)
    return jnp.sum(0.5 * (
        lv_b - lv_a + (var_a + (mu_a - mu_b) ** 2) / var_b - 1.0
    ), axis=-1)


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
    ):
        self.env = env
        self.prior = FrozenLatentPrior.from_dir(prior_dir)
        self.n_joints = n_joints
        self.w_r = float(w_r)
        self.H = int(history_len)

        if self.prior.n_joints != n_joints:
            raise ValueError(
                f"prior n_joints {self.prior.n_joints} != env n_joints {n_joints}"
            )

        # History feature: (q, qdot, prev_action) per frame.
        self.history_dim = 2 * n_joints + n_joints  # = 3 * n_joints

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def observation_size(self):
        # Brax dict-obs convention: dict mapping key -> int (flat dim).
        # We keep proprioception flat (we'll ravel-pytree it on the fly).
        # Caller can override via build_env if needed.
        return {
            "proprioception": self._proprioception_dim,
            "o_history": self.H * self.history_dim,
            "z_target": self.prior.latent_dim,
        }

    @property
    def _proprioception_dim(self):
        # Computed lazily on first reset; cached.
        if not hasattr(self, "_prop_dim_cached"):
            return None  # caller should run reset() once before reading
        return self._prop_dim_cached

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
        prev_action = jnp.zeros((self.n_joints,))

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

        # Build the new flat obs dict
        prop_flat = _flatten_proprioception(state.obs["state"]["proprioception"])
        # Cache proprioception dim for observation_size
        self._prop_dim_cached = int(prop_flat.shape[-1])

        new_obs = {
            "proprioception": prop_flat,
            "o_history": buf.history.reshape(-1),
            "z_target": z_t_mean,
        }

        info = dict(state.info) if state.info else {}
        info["latent_buf"] = buf
        info["r_mimic"] = jnp.float32(1.0)
        info["mimic_kl"] = jnp.float32(0.0)

        # Initial reward = 1.0 (KL from a window to its predicted future is ~0
        # at reset; paper Eq.9 gives r ~= 1)
        return state.replace(
            obs=new_obs,
            reward=jnp.float32(1.0),
            info=info,
        )

    def step(self, state, action):
        new_state = self.env.step(state, action)
        buf: _Buffers = state.info["latent_buf"]
        qpos = new_state.data.qpos
        qvel = new_state.data.qvel
        buf = self._push_frame(buf, qpos, qvel, action)

        z_sim_mean, z_sim_logvar = self._compute_z_sim(buf.motion_window)
        z_t_mean, z_t_logvar = self._compute_z_target(buf.prev_z_mean)
        kl = _kl_diag_gauss(z_t_mean, z_t_logvar, z_sim_mean, z_sim_logvar)
        r_mimic = jnp.exp(-self.w_r * kl)

        buf = buf.replace(prev_z_mean=z_sim_mean, prev_z_logvar=z_sim_logvar)

        prop_flat = _flatten_proprioception(new_state.obs["state"]["proprioception"])
        new_obs = {
            "proprioception": prop_flat,
            "o_history": buf.history.reshape(-1),
            "z_target": z_t_mean,
        }

        info = dict(new_state.info) if new_state.info else {}
        info["latent_buf"] = buf
        info["r_mimic"] = r_mimic
        info["mimic_kl"] = kl

        return new_state.replace(
            obs=new_obs,
            reward=r_mimic,
            info=info,
        )

    # Pass-through for the rest (render etc.)
    def __getattr__(self, name):
        return getattr(self.env, name)
