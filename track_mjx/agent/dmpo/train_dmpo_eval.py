# track_mjx/agent/dmpo/train_dmpo_eval.py
"""Eval helpers for ``train_dmpo``: rollout, vision-sensitivity diagnostic,
and MP4 rendering with binocular vision overlay.

Mirrors the structure of vnl-playground's
``train_highlvl.binocular_vision_policy_params_fn`` (L2345) but slimmed
down to the bits we need for DMPO eval (no eye-condition ablations in
this first cut — those are queued in future.md).
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import logging
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

log = logging.getLogger(__name__)


def compute_rollout_metrics(rollout: list) -> dict:
    """Aggregate per-rollout scalars matching train_highlvl's eval keys.

    Mirrors ``vnl_playground.train_highlvl._compute_rollout_metrics`` so
    DMPO and PPO eval panels are directly comparable in wandb.

    Returns dict with: cumulative_reward, mean_reward_per_step,
    num_episodes, mean_episode_reward, mean_episode_length,
    total_gap_crossings, plus mean/sum/min/max of any per-step
    ``anchor/*`` metrics emitted by the kl-anchor wrapper (e.g.
    ``anchor/r_anchor``, ``anchor/r_task``, ``anchor/action_mse``).
    Reward-component keys absent on healthy DMPO runs simply don't
    appear, preserving backward compatibility for non-kl-anchor entries.
    """
    total_reward = 0.0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    current_ep_reward = 0.0
    current_ep_length = 0
    total_gap_crossings = 0
    # Per-step anchor metric collection. We grab a value for every
    # ``anchor/*`` key that appears anywhere in the rollout's
    # ``state.metrics`` dicts.
    anchor_values: dict[str, list[float]] = {}

    for state in rollout[1:]:  # skip initial reset state
        r = float(np.asarray(state.reward))
        total_reward += r
        current_ep_reward += r
        current_ep_length += 1

        metrics = getattr(state, "metrics", None)
        if metrics is not None:
            gap_bonus = float(np.asarray(metrics.get("rewards/gap_crossing_bonus", 0.0)))
            if gap_bonus > 0:
                total_gap_crossings += 1
            for k in metrics.keys():
                if isinstance(k, str) and k.startswith("anchor/"):
                    try:
                        v = float(np.asarray(metrics[k]))
                    except Exception:
                        continue
                    anchor_values.setdefault(k, []).append(v)

        if float(np.asarray(getattr(state, "done", 0.0))) > 0.5:
            episode_rewards.append(current_ep_reward)
            episode_lengths.append(current_ep_length)
            current_ep_reward = 0.0
            current_ep_length = 0

    if current_ep_length > 0:
        episode_rewards.append(current_ep_reward)
        episode_lengths.append(current_ep_length)

    n_episodes = len(episode_rewards)
    n_steps = max(len(rollout) - 1, 1)
    out = {
        "cumulative_reward": total_reward,
        "mean_reward_per_step": total_reward / n_steps,
        "num_episodes": n_episodes,
        "mean_episode_reward": (
            sum(episode_rewards) / n_episodes if n_episodes > 0 else 0.0
        ),
        "mean_episode_length": (
            sum(episode_lengths) / n_episodes if n_episodes > 0 else 0.0
        ),
        "total_gap_crossings": total_gap_crossings,
    }
    # Anchor metric aggregates. Use a single nesting level (``anchor/<suffix>_<stat>``)
    # so wandb groups them under an "anchor" panel.
    for full_key, vals in anchor_values.items():
        if not vals:
            continue
        suffix = full_key[len("anchor/"):]
        arr = np.asarray(vals, dtype=np.float64)
        out[f"anchor/{suffix}_mean"] = float(arr.mean())
        out[f"anchor/{suffix}_sum"] = float(arr.sum())
        out[f"anchor/{suffix}_min"] = float(arr.min())
        out[f"anchor/{suffix}_max"] = float(arr.max())
    return out


def compute_vision_sensitivity(
    policy_apply: Callable,
    params: Any,
    obs: dict,
    rng: jax.Array,
) -> float:
    """L2 norm of action(real_vision) - action(blank_vision).

    Diagnostic: 0 means the policy completely ignores vision; large
    values mean the policy is using vision to disambiguate identical
    proprio + task obs. Compare against the expected scale of the policy
    output (~1.0 for tanh-bounded actions).
    """
    obs_blank = dict(obs)
    obs_blank["vision"] = jnp.zeros_like(obs["vision"])
    act_real = policy_apply(params, obs).mode()
    act_blank = policy_apply(params, obs_blank).mode()
    return float(jnp.linalg.norm(act_real - act_blank))


def run_eval_rollout_envzero(
    env: Any,
    policy_apply: Callable,
    params: Any,
    rng: jax.Array,
    episode_length: int,
    num_envs: int,
    normalizer_params=None,
) -> tuple[list, list]:
    """Roll out the *batched* eval env for ``episode_length`` steps and
    return env-0's per-step state.

    Implementation: a single ``@jax.jit`` ``lax.scan`` walks the rollout
    on-device, slicing env 0 inside the scan body so the only host
    transfer is the scan's stacked output. This avoids the re-trace +
    fresh allocation pattern that the previous Python-loop version hit
    (each per-step ``env.step`` re-traced and triggered a 41 MiB XLA
    buffer allocation that OOM'd against the training-side cache).

    The render wrapper is locked to ``nworld=num_envs``; rather than
    rebuild a separate single-env stack, we step the batched env inside
    the scan and pull leaf[0] off each new state for rendering.
    Reward / done from env 0 drive termination tracking.

    Args:
        env: the batched, jittable eval env.
        policy_apply: the network's ``apply`` function returning a
            distribution-like object with a ``.mode()`` method.
        params: policy parameters consumed by ``policy_apply``.
        rng: PRNGKey for env reset and scan.
        episode_length: number of scan steps.
        num_envs: leading batch dimension of the env.
        normalizer_params: Optional DMPO normalizer (DictRunningStatisticsState
            or flat RunningStatisticsState). When provided, obs is normalized
            via the same ``_normalize_obs`` dispatch as the training rollout
            before being passed to ``policy_apply``. Default ``None`` preserves
            the legacy behavior of feeding raw obs to the policy.

    Returns:
        rollout: list of CPU-side State pytrees (env 0 only) of length
            ``episode_length + 1`` (initial reset state, then one per
            scan step).
        termination_events: list of ``(frame_idx, reason)`` tuples.
    """
    from track_mjx.agent.dmpo.action_utils import bind
    from track_mjx.agent.dmpo.learner import _normalize_obs

    rng, k_reset = jax.random.split(rng)
    keys = jax.random.split(k_reset, num_envs)
    state = env.reset(keys)

    def _index_zero(pytree):
        # Slice leading axis of every array leaf; pass scalars through.
        return jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and getattr(x, "ndim", 0) > 0 else x,
            pytree,
        )

    @jax.jit
    def _scan_envzero(initial_state, k_scan):
        def body(carry, _):
            st, k = carry
            k, _ = jax.random.split(k)
            if normalizer_params is None:
                obs_for_policy = st.obs
            else:
                obs_for_policy = _normalize_obs(st.obs, normalizer_params)
            raw = jax.vmap(lambda o: policy_apply(params, o).mode())(obs_for_policy)
            bound = bind(raw)
            out = env.step(st, bound)
            new_st = out[0] if isinstance(out, tuple) else out
            env0 = _index_zero(new_st)
            return (new_st, k), env0

        (_, _), traj = jax.lax.scan(
            body, (initial_state, k_scan), None, length=episode_length
        )
        return traj

    rng, k_scan = jax.random.split(rng)
    traj = _scan_envzero(state, k_scan)
    # ``traj`` has leading axis [episode_length] on every array leaf.
    # Touch one leaf to materialise the scan; subsequent slicing is cheap.
    jax.block_until_ready(jax.tree.leaves(traj)[0])

    rollout: list = [jax.device_get(_index_zero(state))]
    for t in range(episode_length):
        rollout.append(
            jax.device_get(
                jax.tree.map(
                    lambda x: x[t] if hasattr(x, "shape") and getattr(x, "ndim", 0) > 0 else x,
                    traj,
                )
            )
        )

    termination_events: list[tuple[int, str]] = []
    for t, st in enumerate(rollout[1:], start=1):
        done = getattr(st, "done", None)
        if done is None:
            continue
        if float(np.asarray(done)) > 0.5:
            termination_events.append((t, "done"))

    # BraxAutoResetWrapper.step swaps the fall-frame data with reset_data when
    # done=True (mujoco_playground/_src/wrapper.py:188-197), so rollout[t]
    # already contains the reset pose. The fade then overlays on the reset
    # frame, which is jarring. We splice in the previous frame's data/obs
    # while preserving ``done=True`` so the renderer shows the last clean
    # pose during the fade. This matches train_highlvl's appearance, where
    # an unwrapped env preserves the actual fall state.
    for t, _ in termination_events:
        if t == 0:
            continue
        prev = rollout[t - 1]
        cur = rollout[t]

        def _swap(x_cur, x_prev):
            return x_prev

        # Replace data + obs from prev; keep cur.done so termination event
        # marker stays valid.
        new_data = jax.tree.map(_swap, cur.data, prev.data)
        new_obs = jax.tree.map(_swap, cur.obs, prev.obs)
        rollout[t] = cur.replace(data=new_data, obs=new_obs)

    return rollout, termination_events


def render_eval_video(
    rollout: list,
    mj_model,
    video_path: str | Path,
    *,
    fps: int = 50,
    height: int = 480,
    width: int = 640,
    camera: str = "close_profile-rodent",
    overlay_vision: bool = True,
    overlay_scale: int = 4,
    hud_config: dict | None = None,
    reward_config: dict | None = None,
    termination_events: list[tuple[int, str]] | None = None,
    eye_qpos_indices: list | None = None,
) -> str:
    """Render a tracking-camera MP4 with full HUD + vision overlay (env 0).

    Delegates to ``vnl_playground.train_highlvl.render_video`` with
    ``use_obs_vision=True`` so the binocular vision strip is composed
    from ``state.obs["vision"]`` directly (no re-render needed). This
    is the **same** rendering pipeline the PPO baseline uses, including:

    - Tracking camera (default ``close_profile-rodent``).
    - Side-by-side L|R binocular vision overlay in the upper-left.
    - HUD in the bottom-left with speed, reward breakdown, cumulative
      reward, torso height, heading, step counter, etc.
      (gated by ``hud_config``).
    - Termination event overlays with logistic fade.

    Args:
        rollout: list of CPU-side State pytrees (one per frame) with
            ``state.obs["vision"]``, ``state.data.qpos/qvel``, and
            ``state.metrics``.
        mj_model: MuJoCo physics model for the renderer.
        video_path: output ``.mp4`` path.
        fps, height, width: video parameters.
        camera: MJC camera name (only used for the default tracking; the
            vnl_render uses a tracking camera that follows the torso body).
        overlay_vision: if False, skip the vision overlay (plain tracking video).
        overlay_scale: kept for API compatibility (vnl_render handles its own scale).
        hud_config: dict gating the HUD (see
            ``vnl-playground/.../config/.../hud:`` for the schema).
        reward_config: dict mapping reward term names → params (used to
            display target_speed in the HUD).
        termination_events: list of ``(frame_idx, reason_string)`` tuples.
        eye_qpos_indices: optional indices into qpos for eye joints
            (used by HUD when ``show_eye_angles`` is enabled).
    """
    # Lazy import to avoid pulling in vnl-playground unless we render.
    from vnl_playground.train_highlvl import render_video as _vnl_render_video

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    mj_data = mujoco.MjData(mj_model)
    try:
        _vnl_render_video(
            rollout=rollout,
            mj_model=mj_model,
            mj_data=mj_data,
            renderer=renderer,
            video_path=str(video_path),
            fps=fps,
            vision_renderer=None,
            right_vision_renderer=None,
            termination_events=termination_events,
            hud_config=hud_config,
            reward_config=reward_config,
            use_obs_vision=overlay_vision,
            eye_qpos_indices=eye_qpos_indices,
        )
    finally:
        renderer.close()
    log.info(
        "Wrote %s (%d frames @ %d fps)", str(video_path), len(rollout), fps
    )
    return str(video_path)
