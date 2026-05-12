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
                          use_qvel: bool,
                          active_joints: jnp.ndarray | None = None) -> jnp.ndarray:
    """Build a single per-frame motion descriptor from (qpos, qvel).

    use_qvel=True  -> m = (p, theta, v, q, qdot)            feat = 3+4+6+nj+nj
    use_qvel=False -> m = (p, theta, q)                     feat = 3+4+nj

    If ``active_joints`` is provided, q (and qdot if used) only includes those
    indices into the [0, n_joints) range — drops dead joints (rat fingers/tail
    vertebrae) that the prior was trained without.
    """
    p = qpos[..., :3]
    theta = qpos[..., 3:7]
    q_full = qpos[..., 7:]
    if active_joints is not None:
        q = jnp.take(q_full, active_joints, axis=-1)
    else:
        q = q_full
    if not use_qvel:
        return jnp.concatenate([p, theta, q], axis=-1)
    v = qvel[..., :6]
    qdot_full = qvel[..., 6:]
    if active_joints is not None:
        qdot = jnp.take(qdot_full, active_joints, axis=-1)
    else:
        qdot = qdot_full
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


def _kl_to_unit_gaussian(mu, lv):
    """KL(N(mu, exp(lv)) || N(0, I)), summed over the latent dim.

    Used as a diagnostic during Phase 2 training: tells us how far the
    encoder's posterior on the sim-policy's trajectories sits from the
    standard Gaussian prior the encoder was regularized toward in Phase 1.
    Phase 3 will sample/synthesize z from N(0, I)-ish space, so π_style's
    rollouts must keep their encoded posteriors roughly there too.
    """
    return 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1.0 - lv, axis=-1)


def _flatten_proprioception(prop) -> jnp.ndarray:
    """Take an arbitrarily-nested mapping/array tree and ravel to (N,)."""
    flat, _ = jax.flatten_util.ravel_pytree(prop)
    return flat


@struct.dataclass
class _Buffers:
    motion_window: jnp.ndarray        # (w, feat_dim) — agent's sim window, RAW
    ref_motion_window: jnp.ndarray    # (w, feat_dim) — reference clip window, RAW
    prev_z_ref_mean: jnp.ndarray      # (latent_dim,) z_ref from previous step
    prev_z_ref_logvar: jnp.ndarray    # (latent_dim,)
    prev_action: jnp.ndarray          # (action_dim,) action from prev step,
                                       # used to compute frame-to-frame action
                                       # jerk metric: ||a_t - a_{t-1}|| per step.
                                       # Surfaces as eval/episode_action_jerk
                                       # in wandb so we can quantify rollout
                                       # smoothness independent of metrics like
                                       # ep_len or KL.


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
        history_len: int = 0,           # accepted for back-compat but unused
        prepare_observation_size: bool = True,
        kl_mode: str = "mean",   # 'mean' = unit-var KL = 0.5 ||mu_t - mu_s||^2
                                 # 'full' = full Gaussian KL (paper-faithful but
                                 # explodes when prior has tight posteriors)
        drop_dead_orientations: bool = False,
        proprio_var_threshold: float = 1e-8,
        use_predictor: bool = True,     # False → z_target = E(ref_window_t)
                                         # directly; bypasses the predictor.
                                         # Use when the v8 predictor is unreliable
                                         # (h=1 MSE 65× worse than copy-last).
        sigma_clamp: float = 0.0,        # If >0, clamp BOTH z_sim and z_target
                                         # logvars to ≤ 2*log(sigma_clamp) before
                                         # computing KL. Use to prevent the
                                         # encoder from inflating posterior σ on
                                         # OOD policy trajectories — without it,
                                         # the policy learns to produce uncertain
                                         # sim windows so KL(target||sim) shrinks
                                         # via the variance term even when μ
                                         # don't match. Set to the σ_max used at
                                         # Phase 1 training (e.g. 0.05).
        branch_kl_weights: list | tuple | None = None,  # per-branch KL weights.
                                         # None  → standard full-z KL (legacy).
                                         # When set, must match the prior's
                                         # branch_latent_dims length. r_mimic
                                         # becomes exp(-w_r·Σ wᵢ·KL_i) where
                                         # KL_i is the KL on the i-th branch's
                                         # z slice. Use to differentially weight
                                         # tracking signals (e.g. boost root
                                         # tracking in 3-way splits where root
                                         # has fewer latent dims than limb).
    ):
        self.env = env
        self.prior = FrozenLatentPrior.from_dir(prior_dir)
        self.n_joints = n_joints
        self.w_r = float(w_r)
        self.kl_mode = str(kl_mode)
        self.sigma_clamp = float(sigma_clamp)
        if self.kl_mode not in ("mean", "full"):
            raise ValueError(f"unknown kl_mode {kl_mode!r}; expected 'mean' or 'full'")

        if self.prior.n_joints != n_joints:
            raise ValueError(
                f"prior n_joints {self.prior.n_joints} != env n_joints {n_joints}"
            )

        self.action_dim = int(env.action_size)
        self.use_predictor = bool(use_predictor)
        self._proprio_keep_indices = None  # None = pass-through

        # Per-branch KL weighting setup. Pre-compute slice boundaries so the
        # per-step JIT path doesn't have to re-derive them.
        self._branch_kl_weights = None
        self._branch_slices = None
        if branch_kl_weights is not None and len(branch_kl_weights) > 0:
            if self.prior.branch_latent_dims is None:
                raise ValueError(
                    "branch_kl_weights requires a prior trained with body-part-"
                    "split heads (branch_latent_dims is None)."
                )
            if len(branch_kl_weights) != len(self.prior.branch_latent_dims):
                raise ValueError(
                    f"branch_kl_weights length {len(branch_kl_weights)} != "
                    f"prior branches {len(self.prior.branch_latent_dims)} "
                    f"(branch_names={self.prior.branch_names})"
                )
            self._branch_kl_weights = tuple(float(x) for x in branch_kl_weights)
            slices, off = [], 0
            for d in self.prior.branch_latent_dims:
                slices.append((off, off + int(d)))
                off += int(d)
            self._branch_slices = tuple(slices)

        # If the prior was trained on a reduced joint set, mirror the same
        # active_joints index array on the JAX side so we mask sim qpos/qvel
        # the same way before feeding them to the encoder.
        if self.prior.active_joints is not None:
            self._active_joints_jax = jnp.asarray(
                self.prior.active_joints, dtype=jnp.int32
            )
        else:
            self._active_joints_jax = None

        # Pre-compute proprioception flat dim by running a dry reset; the Brax
        # PPO trainer queries observation_size BEFORE the first env.reset(), so
        # caching it lazily on first reset is too late for network init.
        if prepare_observation_size:
            try:
                _probe = self.env.reset(jax.random.PRNGKey(0))
                _prop = _probe.obs["state"]["proprioception"]
                _flat, _ = jax.flatten_util.ravel_pytree(_prop)
                self._prop_dim_cached = int(_flat.shape[-1])
                if drop_dead_orientations:
                    self._build_orientation_keep_mask(proprio_var_threshold)
            except Exception as e:  # noqa: BLE001 — keep wrapper usable even if probe fails
                self._prop_dim_cached = None
                import logging as _logging
                _logging.warning(f"LatentMimicEnvWrapper proprioception probe failed: {e}")

    def _build_orientation_keep_mask(self, var_threshold: float):
        """Build a per-dim keep mask that drops dead orientation features.

        Probes the env's proprioception across N reference clip frames (using
        the actual qpos/qvel from the reference), measures per-dim variance,
        and marks dims as 'dead' iff (var < var_threshold) AND (the dim falls
        in the orientations block — first n_orient dims of the flat proprio).
        Velocity, joint_angles, prev_action, height, upright are always kept.

        The walker base proprio layout is:
            orientations (n_orient) | height (1) | upright (1) | qvel (nv)
            | joint_angles (nj) | prev_action (na)
        with n_orient = total - 2 - nv - nj - na.
        """
        import numpy as _np
        import logging as _logging
        from mujoco import mjx as _mjx

        try:
            inner = self.env
            ref_qpos_all = _np.asarray(inner.reference_clips.qpos)  # (n_clips, n_frames, nq)
            ref_qvel_all = _np.asarray(inner.reference_clips.qvel)  # (n_clips, n_frames, nv)
            n_clips, n_frames, _ = ref_qpos_all.shape

            # Subsample frames to keep startup fast.
            stride = max(1, (n_clips * n_frames) // 4000)
            qp = ref_qpos_all.reshape(-1, ref_qpos_all.shape[-1])[::stride]
            qv = ref_qvel_all.reshape(-1, ref_qvel_all.shape[-1])[::stride]

            mj_model = inner.mjx_model
            cfg = inner._config

            @jax.jit
            def proprio_one(qpos, qvel):
                data = _mjx.make_data(
                    inner.mj_model, impl=cfg.mujoco_impl,
                    njmax=cfg.njmax, naconmax=cfg.naconmax,
                )
                data = data.replace(qpos=qpos, qvel=qvel)
                data = _mjx.forward(mj_model, data)
                info = {"prev_action": jnp.zeros(self.action_dim)}
                return inner._get_proprioception(data, info, flatten=True)

            P = jax.vmap(proprio_one)(jnp.asarray(qp), jnp.asarray(qv))
            P = _np.asarray(P)
            var = P.var(axis=0)

            total = P.shape[1]
            nv = mj_model.nv
            nj = mj_model.nq - 7
            na = self.action_dim
            n_orient = total - 2 - nv - nj - na  # height(1) + upright(1)
            assert n_orient >= 0, (
                f"unexpected proprio layout: total={total}, nv={nv}, "
                f"nj={nj}, na={na} → n_orient={n_orient}"
            )

            keep = _np.ones(total, dtype=bool)
            # Only dead dims in orientations are dropped; everything else kept.
            for i in range(n_orient):
                if var[i] < var_threshold:
                    keep[i] = False

            n_dropped = int((~keep).sum())
            self._proprio_keep_indices = jnp.asarray(_np.where(keep)[0], dtype=jnp.int32)
            self._prop_dim_cached = int(keep.sum())
            _logging.info(
                f"[LatentMimicEnvWrapper] orientation pruning: "
                f"{n_orient} orient dims, dropped {n_dropped} dead "
                f"(var<{var_threshold}). proprio: {total} → {self._prop_dim_cached}"
            )
        except Exception as e:  # noqa: BLE001
            _logging.warning(
                f"[LatentMimicEnvWrapper] orientation pruning failed: {e}; "
                f"falling back to full proprio."
            )
            self._proprio_keep_indices = None

    def _project_proprio(self, prop_flat):
        """Apply the keep-mask projection if one is configured."""
        if self._proprio_keep_indices is None:
            return prop_flat
        return jnp.take(prop_flat, self._proprio_keep_indices)

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
        # Conform to track-mjx's existing two-key schema (task_obs, proprioception).
        # task_obs = z_target (60-dim). proprioception is the base env's flat
        # proprioception passed through unchanged — same byte-layout as the
        # previous mimic-mjx intention-arch setup.
        return {
            "state": {
                "task_obs": self.prior.latent_dim,
                "proprioception": self._prop_dim_cached,
            }
        }

    # ------------------------- buffer helpers -------------------------

    def _empty_buffers(self) -> _Buffers:
        return _Buffers(
            motion_window=jnp.zeros((self.prior.window_len, self.prior.feat_dim)),
            ref_motion_window=jnp.zeros((self.prior.window_len, self.prior.feat_dim)),
            prev_z_ref_mean=jnp.zeros((self.prior.latent_dim,)),
            prev_z_ref_logvar=jnp.zeros((self.prior.latent_dim,)),
            prev_action=jnp.zeros((self.action_dim,)),
        )

    def _push_frame(self, buf: _Buffers, qpos, qvel) -> _Buffers:
        m = _frame_from_qpos_qvel(qpos, qvel, self.prior.use_qvel,
                                  self._active_joints_jax)
        new_motion = jnp.concatenate(
            [buf.motion_window[1:], m[None, :]], axis=0
        )
        return buf.replace(motion_window=new_motion)

    def _push_ref_frame(self, buf: _Buffers, ref_qpos, ref_qvel) -> _Buffers:
        """Push a reference-clip frame onto the reference motion window."""
        m = _frame_from_qpos_qvel(ref_qpos, ref_qvel, self.prior.use_qvel,
                                  self._active_joints_jax)
        new_ref = jnp.concatenate(
            [buf.ref_motion_window[1:], m[None, :]], axis=0
        )
        return buf.replace(ref_motion_window=new_ref)

    def _ref_qpos_qvel(self, clip_idx, frame):
        """Read (qpos, qvel) for one reference frame, clamped to clip bounds.

        clip_idx and frame may be traced JAX scalars. Clamping keeps us safe
        when the env is at a truncation boundary (cur_frame can momentarily
        exceed valid range). ReferenceClips.at uses simple advanced indexing,
        which JAX handles via gather under vmap.
        """
        n_frames = self.env.reference_clips.qpos.shape[1]
        f = jnp.clip(jnp.asarray(frame, dtype=jnp.int32), 0, n_frames - 1)
        ref = self.env.reference_clips.at(clip=clip_idx, frame=f)
        return ref.qpos, ref.qvel

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

        # Reference frame at the env's start_frame (sim is reset to match this).
        clip_idx = state.info["reference_clip"]
        start_frame = state.info["start_frame"]
        ref_qpos0, ref_qvel0 = self._ref_qpos_qvel(clip_idx, start_frame)

        buf = self._empty_buffers()
        # Fill both windows with the initial pose so the encoder sees a sensible
        # (non-zero) window from the first step. At reset, sim qpos == ref qpos.
        for _ in range(self.prior.window_len):
            buf = self._push_frame(buf, qpos, qvel)
            buf = self._push_ref_frame(buf, ref_qpos0, ref_qvel0)

        # z_ref is the latent encoding of the reference motion window.
        z_ref_mean, z_ref_logvar = self._compute_z_sim(buf.ref_motion_window)
        buf = buf.replace(prev_z_ref_mean=z_ref_mean, prev_z_ref_logvar=z_ref_logvar)
        if self.use_predictor:
            z_t_mean, z_t_logvar = self._compute_z_target(buf.prev_z_ref_mean)
        else:
            # Bypass the (broken) predictor — target the encoded reference
            # window directly. Reward becomes "match reference latent state".
            z_t_mean, z_t_logvar = z_ref_mean, z_ref_logvar

        # Take base env's proprioception, optionally drop dead orientation dims.
        prop_flat = _flatten_proprioception(state.obs["state"]["proprioception"])
        if not hasattr(self, "_prop_dim_cached") or self._prop_dim_cached is None:
            self._prop_dim_cached = int(prop_flat.shape[-1])
        prop_flat = self._project_proprio(prop_flat)

        from collections import OrderedDict
        new_obs = OrderedDict(
            state=OrderedDict(
                task_obs=z_t_mean,
                proprioception=prop_flat,
            )
        )

        info = dict(state.info) if state.info else {}
        info["latent_buf"] = buf
        info["r_mimic"] = jnp.float32(1.0)
        info["mimic_kl"] = jnp.float32(0.0)
        # action_jerk placeholder for pytree-consistency between reset and step.
        info["action_jerk"] = jnp.float32(0.0)

        metrics = dict(state.metrics) if state.metrics else {}
        metrics["r_mimic"] = jnp.float32(1.0)
        metrics["mimic_kl"] = jnp.float32(0.0)
        metrics["sim_ref_l2"] = jnp.float32(0.0)
        metrics["action_jerk"] = jnp.float32(0.0)
        # Posterior diagnostics for Phase 3 readiness. At reset, sim qpos == ref
        # qpos so z_sim_post == z_ref_post. Use the just-computed z_ref values
        # as the seed.
        metrics["sigma_sim_mean"] = jnp.mean(jnp.exp(0.5 * z_ref_logvar))
        metrics["kl_sim_to_prior"] = _kl_to_unit_gaussian(z_ref_mean, z_ref_logvar)
        # Per-branch KL placeholders MUST be initialized here for jax.lax.scan
        # in Brax's training loop — the scan body requires identical pytree
        # structure between reset (carry input) and step (carry output).
        if self._branch_slices is not None and self.prior.branch_names is not None:
            for bname in self.prior.branch_names:
                metrics[f"mimic_kl_{bname}"] = jnp.float32(0.0)
                info[f"mimic_kl_{bname}"] = jnp.float32(0.0)

        return state.replace(
            obs=new_obs,
            reward=jnp.float32(1.0),
            info=info,
            metrics=metrics,
        )

    def step(self, state, action):
        new_state = self.env.step(state, action)
        buf: _Buffers = state.info["latent_buf"]

        # Frame-to-frame action jerk (per-step ||a_t - a_{t-1}||). At episode
        # start prev_action is zero, so the first step's jerk is just ||a_0||
        # which is fine — averaged over an episode it converges to the steady-
        # state jerk of the policy. Logged as eval/episode_action_jerk via
        # state.metrics so we can quantify rollout smoothness independent of
        # KL/r/ep_len. Lower is smoother.
        action_jerk = jnp.linalg.norm(action - buf.prev_action)
        buf = buf.replace(prev_action=action)

        # Push agent's new frame onto the sim window.
        sim_qpos = new_state.data.qpos
        sim_qvel = new_state.data.qvel
        buf = self._push_frame(buf, sim_qpos, sim_qvel)

        # Push the reference clip's frame at the new current_frame onto the
        # reference window. This anchors z_target to the reference motion
        # rather than to the agent's own past (which gave no learning signal).
        clip_idx = state.info["reference_clip"]
        cur_frame = jnp.asarray(new_state.metrics["current_frame"], dtype=jnp.int32)
        ref_qpos, ref_qvel = self._ref_qpos_qvel(clip_idx, cur_frame)
        buf = self._push_ref_frame(buf, ref_qpos, ref_qvel)

        # Encode both windows (same encoder, same normalizer).
        z_sim_mean, z_sim_logvar = self._compute_z_sim(buf.motion_window)
        z_ref_mean, z_ref_logvar = self._compute_z_sim(buf.ref_motion_window)

        if self.use_predictor:
            # z_target = E(P(prev_z_ref)) — paper formulation.
            z_t_mean, z_t_logvar = self._compute_z_target(buf.prev_z_ref_mean)
        else:
            # Bypass predictor: target the current reference latent directly.
            z_t_mean, z_t_logvar = z_ref_mean, z_ref_logvar

        # Optional σ clamp before KL: prevents the encoder from inflating
        # posterior σ on OOD sim trajectories, which lets the policy game
        # KL(target||sim) via the variance term instead of matching μ.
        # Apply symmetrically (sim AND target) so the KL formula sees a
        # bounded posterior on both sides — preserves paper semantics on
        # in-distribution data while closing the OOD escape hatch.
        if self.sigma_clamp > 0.0:
            cap = 2.0 * jnp.log(jnp.asarray(self.sigma_clamp, dtype=z_sim_logvar.dtype))
            z_sim_logvar_kl = jnp.minimum(z_sim_logvar, cap)
            z_t_logvar_kl = jnp.minimum(z_t_logvar, cap)
        else:
            z_sim_logvar_kl = z_sim_logvar
            z_t_logvar_kl = z_t_logvar
        # Per-dim KL contributions, then either sum-over-dims (legacy full-z)
        # or weighted-sum-over-branches (when branch_kl_weights is set).
        # Both paths produce the same shape scalar; per-branch path costs no
        # extra encoder calls — just slices the existing μ/logvar.
        if self._branch_slices is not None:
            kl_terms = []
            for w_b, (s, e) in zip(self._branch_kl_weights, self._branch_slices):
                if self.kl_mode == "full":
                    kl_b = _kl_diag_gauss(
                        z_t_mean[s:e], z_t_logvar_kl[s:e],
                        z_sim_mean[s:e], z_sim_logvar_kl[s:e],
                    )
                else:
                    kl_b = _mean_kl(z_t_mean[s:e], z_sim_mean[s:e])
                kl_terms.append(kl_b)
            # Weighted sum drives the reward; unweighted sum is the diagnostic
            # `mimic_kl` we log so it stays comparable across runs.
            weighted_kl = sum(w * k for w, k in zip(self._branch_kl_weights, kl_terms))
            kl = sum(kl_terms)
            r_mimic = jnp.exp(-self.w_r * weighted_kl)
            kl_per_branch = jnp.stack(kl_terms)   # (n_branches,)
        else:
            if self.kl_mode == "full":
                kl = _kl_diag_gauss(z_t_mean, z_t_logvar_kl, z_sim_mean, z_sim_logvar_kl)
            else:
                kl = _mean_kl(z_t_mean, z_sim_mean)
            r_mimic = jnp.exp(-self.w_r * kl)
            kl_per_branch = None

        buf = buf.replace(prev_z_ref_mean=z_ref_mean, prev_z_ref_logvar=z_ref_logvar)

        prop_flat = _flatten_proprioception(new_state.obs["state"]["proprioception"])
        prop_flat = self._project_proprio(prop_flat)

        from collections import OrderedDict
        new_obs = OrderedDict(
            state=OrderedDict(
                task_obs=z_t_mean,
                proprioception=prop_flat,
            )
        )

        # Diagnostic: how far apart are sim and ref encodings before the
        # predictor? This is the "raw" matching signal independent of P.
        sim_ref_l2 = jnp.sqrt(jnp.sum((z_sim_mean - z_ref_mean) ** 2))

        info = dict(new_state.info) if new_state.info else {}
        info["latent_buf"] = buf
        info["r_mimic"] = r_mimic
        info["mimic_kl"] = kl
        info["action_jerk"] = action_jerk
        # Mirror per-branch KL into info so the custom rollout logger can
        # surface per-branch curves at eval time. (state.metrics carries the
        # SAME values for the Brax PPO evaluator's episode aggregates.)
        if kl_per_branch is not None and self.prior.branch_names is not None:
            for i, bname in enumerate(self.prior.branch_names):
                info[f"mimic_kl_{bname}"] = kl_per_branch[i]

        # Promote r_mimic / mimic_kl into metrics so brax PPO's eval evaluator
        # picks them up as eval/episode_rewards/r_mimic etc., visible in wandb
        # at every eval (not just at the policy_params_fn callback every 10M).
        # Phase-3-readiness diagnostics: how far is z_sim's posterior from the
        # standard Gaussian prior the encoder was regularized toward? If σ_sim
        # collapses or KL_to_prior explodes during Phase 2, π_style is
        # producing trajectories that don't lie on the prior's manifold —
        # bad for Phase 3 latent control.
        sigma_sim_mean = jnp.mean(jnp.exp(0.5 * z_sim_logvar))
        kl_sim_to_prior = _kl_to_unit_gaussian(z_sim_mean, z_sim_logvar)

        metrics = dict(new_state.metrics) if new_state.metrics else {}
        metrics["r_mimic"] = r_mimic
        metrics["mimic_kl"] = kl
        metrics["sim_ref_l2"] = sim_ref_l2
        metrics["sigma_sim_mean"] = sigma_sim_mean
        metrics["kl_sim_to_prior"] = kl_sim_to_prior
        metrics["action_jerk"] = action_jerk
        # Per-branch KL diagnostics so we can see which branch dominates and
        # whether the policy is actually getting differential gradient on the
        # boosted branch (typically root in 3-way runs).
        if kl_per_branch is not None and self.prior.branch_names is not None:
            for i, bname in enumerate(self.prior.branch_names):
                metrics[f"mimic_kl_{bname}"] = kl_per_branch[i]

        return new_state.replace(
            obs=new_obs,
            reward=r_mimic,
            info=info,
            metrics=metrics,
        )

    # Pass-through for the rest (render etc.)
    def __getattr__(self, name):
        return getattr(self.env, name)
