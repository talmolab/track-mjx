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

        # Gap crossings come from state.info, not from the reward metrics. See
        # compute_batch_rollout_metrics: `rewards/gap_crossing_bonus` exists only
        # when that term is enabled in env_config.reward_terms, which the
        # frozen-prior arms deliberately switch off -- so reading the bonus
        # reported 0 crossings unconditionally on exactly the arms being studied,
        # while the batch metric (reading info) measured 2.89 per env on the same
        # rollout. The reward bonus stays as a fallback for configs that enable it.
        info = getattr(state, "info", None)
        if isinstance(info, dict) and "just_crossed_gap" in info:
            if bool(np.any(np.asarray(info["just_crossed_gap"]))):
                total_gap_crossings += 1

        metrics = getattr(state, "metrics", None)
        if metrics is not None:
            if not (isinstance(info, dict) and "just_crossed_gap" in info):
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
            # ALL-ENV statistics. The scan already steps every env; keeping only
            # env 0 threw away 2047/2048 of the sample and made
            # mean_episode_length == episode_length/n_episodes with n a small
            # integer (sd 35.3 over the baseline's own flat window). These three
            # [num_envs] vectors are what make the eval an estimator.
            allenv = {"reward": new_st.reward, "done": new_st.done}
            # Carry every per-step `rewards/*` and `terminations/*` env metric
            # through for ALL envs. brax's EvalWrapper sums exactly these over
            # the episode and reports them as `eval/episode_<key>`, which is how
            # the PPO reference produced eval/episode_rewards/forward_velocity
            # and eval/episode_terminations/fallen. Carrying them here is what
            # lets DMPO emit the identical keys.
            m = getattr(new_st, "metrics", None)
            if isinstance(m, dict):
                for mk, mv in m.items():
                    if isinstance(mk, str) and (
                        mk.startswith("rewards/")
                        or mk.startswith("terminations/")
                        # `anchor/*` added 2026-08-18. These were previously
                        # reported from the env-0 path ONLY, i.e. n=1. At h1's
                        # 253.1M eval that produced an apparent regime shift --
                        # r_task 0.847 -> 0.517, r_anchor 0.294 -> 0.505 -- which
                        # fully reverted one eval later; the all-env
                        # `reward_per_step` in the same log line barely moved. It
                        # was one rat having a bad episode. `anchor/r_anchor` is
                        # the only continuous readout of drift from the frozen
                        # prior, so it cannot stay the least reliable number we
                        # log -- it is the primary measurement for the decoder-thaw
                        # arm.
                        or mk.startswith("anchor/")
                    ):
                        arr = jnp.asarray(mv, dtype=new_st.reward.dtype)
                        # Only per-env scalars are aggregatable. Anything else
                        # (a per-joint vector, say) is skipped rather than
                        # allowed to raise inside the jitted scan, which would
                        # take down the whole training run at its first eval.
                        if arr.shape == new_st.reward.shape or arr.ndim == 0:
                            allenv[mk] = jnp.broadcast_to(arr, new_st.reward.shape)
            # Gap crossings must come from state.info, NOT from the reward
            # metrics above. `rewards/gap_crossing_bonus` only exists when
            # `gap_crossing_bonus` is listed in env_config.reward_terms, and every
            # frozen-prior arm deliberately DROPS that term to keep the reward
            # velocity-only and matched to the PPO reference. The old code read
            # the bonus with a `.get(..., zeros)` default, so on exactly those
            # arms it reported 0.000 crossings unconditionally -- a metric that
            # could not ever be nonzero, reported as if it were a measurement.
            # `info["just_crossed_gap"]` is maintained by the task on every step
            # regardless of the reward configuration (run_gap.py:542-546).
            inf = getattr(new_st, "info", None)
            if isinstance(inf, dict) and "just_crossed_gap" in inf:
                jc = jnp.asarray(inf["just_crossed_gap"], dtype=new_st.reward.dtype)
                if jc.shape == new_st.reward.shape or jc.ndim == 0:
                    allenv["info/just_crossed_gap"] = jnp.broadcast_to(
                        jc, new_st.reward.shape
                    )
            return (new_st, k), (env0, allenv)

        (_, _), (traj, allenv) = jax.lax.scan(
            body, (initial_state, k_scan), None, length=episode_length
        )
        return traj, allenv

    rng, k_scan = jax.random.split(rng)
    traj, allenv = _scan_envzero(state, k_scan)
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

    return rollout, termination_events, compute_batch_rollout_metrics(allenv)


def compute_batch_rollout_metrics(allenv: dict) -> dict:
    """Proper episode statistics over ALL envs, from the [T, N] scan output.

    Replaces the env-0 estimator, whose `mean_episode_length` was identically
    `episode_length / n_episodes` for a small integer n -- on the baseline's own
    flat 31.3M-105.9M window that gave 55.6, 62.5, 83.3, 100.0, 62.5, 166.7,
    83.3, 71.4: mean 87.7, sd 35.3, a 3x spread while every training metric was
    constant. Any arm smaller than ~2x was unmeasurable.

    Two statistics are reported because they answer different questions:

    `batch/mean_episode_length`  -- COMPLETE episodes only, pooled over envs.
        Unbiased for the fall rate. Excludes the trailing partial episode; the
        old code appended it, which is what forced the sum to equal T exactly.

    `batch/first_episode_length` -- first episode per env, censored at T.
        This is what brax's Evaluator reports, so it is the only number
        comparable to PPO's `eval/avg_episode_length`. Still censored, so it is
        a LOWER bound once episodes approach T.
    """
    rew = np.asarray(allenv["reward"])          # [T, N]
    done = np.asarray(allenv["done"]) > 0.5     # [T, N]
    T, N = rew.shape
    # Prefer the task's own crossing flag over the reward bonus. The bonus is
    # absent whenever `gap_crossing_bonus` is not in env_config.reward_terms --
    # which is EVERY frozen-prior arm, by design, to keep the reward
    # velocity-only and matched to the PPO reference. Reading the bonus with a
    # zeros default therefore reported "0.000 crossings" as a measurement when it
    # was really "this metric is not wired up on this config". `gap_measured`
    # records which source was used so a zero can be told apart from a blind spot.
    if "info/just_crossed_gap" in allenv:
        gap = np.asarray(allenv["info/just_crossed_gap"])
        gap_measured = 1.0
    elif "rewards/gap_crossing_bonus" in allenv:
        gap = np.asarray(allenv["rewards/gap_crossing_bonus"])
        gap_measured = 1.0
    else:
        gap = np.zeros_like(rew)
        gap_measured = 0.0

    comp_len: list[int] = []
    comp_rew: list[float] = []
    first_len = np.full(N, T, dtype=np.int64)
    first_done = np.zeros(N, dtype=bool)

    for e in range(N):
        idx = np.flatnonzero(done[:, e])
        if idx.size:
            first_len[e] = idx[0] + 1
            first_done[e] = True
        start = 0
        for d in idx:
            comp_len.append(int(d - start + 1))
            comp_rew.append(float(rew[start:d + 1, e].sum()))
            start = d + 1
        # trailing partial episode deliberately DROPPED

    out = {
        "batch/num_envs": float(N),
        "batch/steps": float(T),
        "batch/reward_per_step": float(rew.mean()),
        "batch/n_complete_episodes": float(len(comp_len)),
        "batch/mean_episode_length": float(np.mean(comp_len)) if comp_len else float(T),
        "batch/episode_length_sem": (
            float(np.std(comp_len, ddof=1) / np.sqrt(len(comp_len))) if len(comp_len) > 1 else 0.0
        ),
        "batch/mean_episode_reward": float(np.mean(comp_rew)) if comp_rew else 0.0,
        "batch/first_episode_length": float(first_len.mean()),
        "batch/first_episode_length_sem": float(first_len.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0,
        "batch/frac_envs_terminated": float(first_done.mean()),
        # Rate, not a count: the env-0 count was 0 at all 28 baseline evals, so
        # it could not distinguish "never crosses" from "crosses rarely".
        # NOTE the denominators differ, and the difference matters. An env runs
        # SEVERAL episodes inside the T-step window (T/mean_episode_length of
        # them), so `per_env` is a per-WINDOW figure and reads several times
        # larger than the per-attempt rate. `per_episode` is the interpretable
        # one: crossings per episode actually attempted.
        "batch/gap_crossings_per_env": float((gap > 0).sum() / N),
        "batch/gap_crossings_per_episode": (
            float((gap > 0).sum() / len(comp_len)) if comp_len else 0.0
        ),
        "batch/frac_envs_crossing_gap": float(((gap > 0).any(axis=0)).mean()),
        # 1.0 = the two keys above came from a real signal; 0.0 = neither source
        # was present and they are structurally zero. Treat a 0 crossing rate as
        # meaningful ONLY when this is 1.0.
        "batch/gap_measured": gap_measured,
    }

    # All-env anchor diagnostics, alongside the legacy env-0 `anchor/*` keys
    # (kept so older runs stay comparable). SEM over the [T, N] pool is reported
    # because the whole reason these moved here is that the n=1 version could not
    # tell a real drift from one bad episode.
    for k, v in allenv.items():
        if not k.startswith("anchor/"):
            continue
        a = np.asarray(v, dtype=np.float64)
        out[f"batch/{k}"] = float(a.mean())
        out[f"batch/{k}_sem"] = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0

    # ---- PPO-CONVENTION KEYS -------------------------------------------------
    # brax's EvalWrapper accumulates state.metrics over the FIRST episode of each
    # env (censored at the rollout cap), averages over envs, and prefixes
    # `eval/episode_`. Reproducing that exactly is the only way DMPO's numbers can
    # be put on the same axis as the PPO reference's. Note the semantic trap this
    # removes: DMPO's native `cumulative_reward` is the sum over the WHOLE rollout
    # (all episodes), whereas PPO's `episode_reward` is a PER-EPISODE sum -- two
    # similarly-named quantities that differ by a factor of n_episodes.
    alive = np.ones((T, N), dtype=bool)
    if done.any():
        # steps strictly after the first done are outside episode 1
        alive = np.cumsum(done, axis=0) - done.astype(int) == 0

    def _first_ep_sum(x):
        return (np.asarray(x) * alive).sum(axis=0)   # [N]

    # Keys are UNPREFIXED on purpose: the entry point logs these as
    # `wandb.log({f"eval/{k}": v ...})`, so `episode_reward` lands as
    # `eval/episode_reward` -- byte-identical to the PPO reference's key.
    ep_rew = _first_ep_sum(rew)
    out["episode_reward"] = float(ep_rew.mean())
    out["episode_reward_std"] = float(ep_rew.std())
    out["avg_episode_length"] = float(alive.sum(axis=0).mean())
    for k, v in allenv.items():
        if not isinstance(k, str):
            continue
        if k.startswith("rewards/") or k.startswith("terminations/"):
            s = _first_ep_sum(v)
            out[f"episode_{k}"] = float(s.mean())
            out[f"episode_{k}_std"] = float(s.std())
    return out


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
