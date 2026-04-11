"""Experiment: RNN Hidden State Dynamics.

Analyzes per-code kinematics in the reference data to find the most
representative code for each behavior type (walking, rearing, immobility),
then holds each code for 250 frames with K bodies to collect hidden states,
qpos, and render verification ghost videos.

Usage:
    cd moseq_jax
    python -m experiments.run_hidden_dynamics
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from moseq_env_wrapper import MoSeqImitation

from experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    make_inference_fn,
)
from experiments.shared.clip_selection import load_balanced_splits
from experiments.shared.ghost_rendering import (
    build_ghost_model,
    render_ghost_video,
)
from experiments.shared.plotting import get_trajectory_colors

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-code kinematic analysis on reference data
# ---------------------------------------------------------------------------


def analyze_code_kinematics(
    ref_qpos: np.ndarray,
    all_codes: np.ndarray,
    min_bout_frames: int = 5,
) -> dict[int, dict]:
    """Compute per-code kinematic signatures from reference data.

    For each KPMS code, extracts all bouts across all clips and computes:
      - z_rise: mean change in root Z from bout start to end
      - xy_disp: mean total XY path length during bout
      - xyz_disp: mean endpoint displacement in XYZ

    Args:
        ref_qpos: ``[n_clips, clip_len, nq]`` reference joint positions.
        all_codes: ``[n_clips, clip_len]`` KPMS code assignments.
        min_bout_frames: Ignore bouts shorter than this.

    Returns:
        ``{code_id: {n_bouts, z_rise_mean, xy_disp_mean, xyz_disp_mean}}``
    """
    n_clips, clip_len = all_codes.shape
    unique_codes = np.unique(all_codes)
    stats = {}

    for code_id in unique_codes:
        z_rises, xy_disps, xyz_disps = [], [], []

        for ci in range(n_clips):
            seq = all_codes[ci]
            qp = ref_qpos[ci]

            # Find contiguous bouts of this code
            mask = seq == code_id
            if not mask.any():
                continue

            # Extract bout boundaries
            diff = np.diff(mask.astype(int))
            starts = list(np.where(diff == 1)[0] + 1)
            if mask[0]:
                starts.insert(0, 0)
            ends = list(np.where(diff == -1)[0] + 1)
            if mask[-1]:
                ends.append(clip_len)

            for s, e in zip(starts, ends):
                if e - s < min_bout_frames:
                    continue
                bout = qp[s:e]
                z_rises.append(float(bout[-1, 2] - bout[0, 2]))
                xy = bout[:, :2]
                xy_disps.append(
                    float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
                )
                xyz_disps.append(
                    float(np.linalg.norm(bout[-1, :3] - bout[0, :3]))
                )

        if z_rises:
            stats[int(code_id)] = {
                "n_bouts": len(z_rises),
                "z_rise_mean": float(np.mean(z_rises)),
                "xy_disp_mean": float(np.mean(xy_disps)),
                "xyz_disp_mean": float(np.mean(xyz_disps)),
            }

    return stats


def select_behavior_codes(
    code_stats: dict[int, dict],
    min_bouts: int = 3,
) -> dict[str, int]:
    """Select the best code for each behavior type.

    Args:
        code_stats: Output of :func:`analyze_code_kinematics`.
        min_bouts: Minimum number of bouts to consider a code.

    Returns:
        ``{"walk": code_id, "rear": code_id, "immobility": code_id}``
    """
    eligible = {
        c: s for c, s in code_stats.items() if s["n_bouts"] >= min_bouts
    }

    walk_code = max(eligible, key=lambda c: eligible[c]["xy_disp_mean"])
    rear_code = max(eligible, key=lambda c: eligible[c]["z_rise_mean"])
    imm_code = min(eligible, key=lambda c: eligible[c]["xyz_disp_mean"])

    return {"walk": walk_code, "rear": rear_code, "immobility": imm_code}


# ---------------------------------------------------------------------------
# Custom rollout that captures hidden states + qpos
# ---------------------------------------------------------------------------


def run_rollout_with_hidden(
    env,
    inference_fn,
    params,
    ppo_networks,
    key,
    max_steps: int = 250,
    code_override: np.ndarray | None = None,
    reset_clip_idx: int | None = None,
    jit_reset=None,
    jit_step=None,
) -> dict[str, np.ndarray]:
    """Run a single rollout collecting GRU hidden states and qpos."""
    if jit_reset is None:
        jit_reset = jax.jit(env.reset)
    if jit_step is None:
        jit_step = jax.jit(env.step)

    key, reset_key = jax.random.split(key)

    if reset_clip_idx is not None:
        state = env.reset(reset_key, clip_idx=jnp.int32(reset_clip_idx))
    else:
        state = jit_reset(reset_key)

    hidden = ppo_networks.policy_network.init_hidden(1)
    _code_stack_size = int(state.obs["kpms_code"].shape[-1])

    hidden_list: list[np.ndarray] = []
    code_list: list[int] = []
    qpos_list: list[np.ndarray] = []
    reward_list: list[float] = []
    survival = max_steps

    for t in range(max_steps):
        if code_override is not None:
            stacked = []
            for si in range(_code_stack_size):
                idx = min(t + si, len(code_override) - 1)
                stacked.append(float(code_override[idx]))
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = jnp.array(stacked, dtype=jnp.float32)
            state = state.replace(obs=new_obs)

        key, subkey = jax.random.split(key)
        batched_obs = jax.tree.map(lambda x: x[None], state.obs)
        action, extras, hidden = inference_fn(
            params, batched_obs, hidden, subkey,
        )
        action = jax.tree.map(lambda x: x[0], action)
        extras = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
            extras,
        )

        hidden_list.append(np.array(hidden[-1][0]))

        if "code_idx" in extras:
            code_list.append(int(extras["code_idx"]))
        elif "indices" in extras:
            code_list.append(int(extras["indices"]))

        qpos_list.append(np.array(state.data.qpos))

        state = jit_step(state, action)
        reward_list.append(float(state.reward))

        if state.done and survival == max_steps:
            survival = t + 1

    return {
        "hidden_states": np.array(hidden_list),
        "code_indices": np.array(code_list),
        "qpos": np.array(qpos_list),
        "rewards": np.array(reward_list),
        "survival": survival,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="hidden_dynamics_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== RNN Hidden State Dynamics Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path,
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    if not use_rnn:
        raise ValueError("This experiment requires an RNN decoder checkpoint.")

    params = (norm_state, policy_params)

    # ------------------------------------------------------------------
    # Load codes and reference data
    # ------------------------------------------------------------------
    codes_data = np.load(cfg.data.codes_path)
    all_codes = codes_data["all_codes"]  # [n_clips, clip_len]
    test_codes = codes_data["test_codes"]
    train_idx = codes_data["train_indices"]
    test_idx = codes_data["test_indices"]
    all_idx = np.concatenate([train_idx, test_idx])

    splits = load_balanced_splits(cfg.data.balanced_split_path)
    test_indices = splits["balanced"]["test_indices"]

    # Load reference qpos for kinematic analysis
    clip_len = int(ckpt_cfg.env_config.clip_length)
    with h5py.File(cfg.data.reference_data_path, "r") as h5:
        qpos_flat = h5["qpos"][:]
    n_total_clips = qpos_flat.shape[0] // clip_len
    qpos_all = qpos_flat[: n_total_clips * clip_len].reshape(
        n_total_clips, clip_len, -1,
    )
    ref_qpos = qpos_all[all_idx]  # [484, 250, 74]

    # ------------------------------------------------------------------
    # Analyze per-code kinematics → select 3 codes
    # ------------------------------------------------------------------
    log.info("Analyzing per-code kinematics in reference data...")
    code_stats = analyze_code_kinematics(ref_qpos, all_codes)

    log.info(
        f"{'Code':>4} {'Bouts':>6} {'Z Rise':>10} {'XY Disp':>10} "
        f"{'XYZ Disp':>10}"
    )
    for c in sorted(code_stats.keys()):
        s = code_stats[c]
        log.info(
            f"{c:>4} {s['n_bouts']:>6} {s['z_rise_mean']:>10.4f} "
            f"{s['xy_disp_mean']:>10.4f} {s['xyz_disp_mean']:>10.4f}"
        )

    min_bouts = int(cfg.get("min_bouts", 3))
    selected_codes = select_behavior_codes(code_stats, min_bouts=min_bouts)

    for beh, code_id in selected_codes.items():
        s = code_stats[code_id]
        log.info(
            f"  {beh}: code {code_id} "
            f"(z_rise={s['z_rise_mean']:.4f}, "
            f"xy_disp={s['xy_disp_mean']:.4f}, "
            f"xyz_disp={s['xyz_disp_mean']:.4f}, "
            f"n_bouts={s['n_bouts']})"
        )

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    test_clips = ReferenceClips(
        data_path=cfg.data.reference_data_path,
        n_frames_per_clip=clip_len,
        keep_clips_idx=np.array(test_indices),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg,
        clips=test_clips,
        kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inf_fn = make_inference_fn(ppo_networks, use_rnn=True, deterministic=True)

    n_test_clips = test_clips.qpos.shape[0]
    log.info(f"Test clips: {n_test_clips}")

    # ------------------------------------------------------------------
    # Run rollouts: 3 codes × K bodies
    # ------------------------------------------------------------------
    K = int(cfg.K)
    seed = int(cfg.seed)
    max_steps = int(cfg.max_steps)

    save_dict: dict[str, np.ndarray] = {}
    save_dict["selected_codes"] = np.array(
        [(beh, code_id) for beh, code_id in selected_codes.items()],
        dtype=object,
    )

    for beh, code_id in selected_codes.items():
        code_seq = np.full(max_steps, code_id, dtype=np.int32)

        beh_hidden: list[np.ndarray] = []
        beh_qpos: list[np.ndarray] = []
        beh_codes: list[np.ndarray] = []
        beh_rewards: list[np.ndarray] = []

        log.info(
            f"\nRunning {K} rollouts for {beh} (code {code_id}, "
            f"{max_steps} frames)..."
        )

        # Random starting clips for each body
        rng = np.random.RandomState(seed)
        start_clips = rng.choice(n_test_clips, size=K, replace=False)

        for ki in range(K):
            key = jax.random.PRNGKey(seed + ki * 1000)
            result = run_rollout_with_hidden(
                env, inf_fn, params, ppo_networks, key,
                max_steps=max_steps,
                code_override=code_seq,
                reset_clip_idx=int(start_clips[ki]),
                jit_reset=jit_reset, jit_step=jit_step,
            )
            beh_hidden.append(result["hidden_states"])
            beh_qpos.append(result["qpos"])
            beh_codes.append(result["code_indices"])
            beh_rewards.append(result["rewards"])
            log.info(
                f"  body {ki}: survival={result['survival']}/{max_steps}, "
                f"mean_reward={result['rewards'].mean():.1f}"
            )

        save_dict[f"hidden_{beh}"] = np.stack(beh_hidden)
        save_dict[f"qpos_{beh}"] = np.stack(beh_qpos)
        save_dict[f"codes_{beh}"] = np.stack(beh_codes)
        save_dict[f"rewards_{beh}"] = np.stack(beh_rewards)

        # ----------------------------------------------------------
        # Render ghost video
        # ----------------------------------------------------------
        traj_colors = get_trajectory_colors(K)

        log.info(f"  Rendering ghost video ({K} bodies)...")
        ghost_model, base_nq = build_ghost_model(
            env,
            num_ghosts=K - 1,
            ghost_colors=traj_colors[1:],
            camera_distance=0.8,
            camera_elevation=-30.0,
            camera_azimuth=135.0,
        )

        video_path = data_dir / f"verify_{beh}_code{code_id}.mp4"
        render_ghost_video(
            ghost_model=ghost_model,
            base_nq=base_nq,
            trajectories_qpos=[q for q in beh_qpos],
            code_sequences=[c for c in beh_codes],
            trajectory_colors=traj_colors,
            output_path=video_path,
            title=f"{beh} (code {code_id}, n={K})",
        )
        log.info(f"  Saved: {video_path}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = data_dir / "hidden_dynamics.npz"
    np.savez_compressed(out_path, **save_dict)
    log.info(f"Saved: {out_path}")

    fig_data_dir = MOSEQ_DIR / "figures" / "data"
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_data_dir / "hidden_dynamics.npz"
    shutil.copy2(out_path, dst)
    log.info(f"Copied to: {dst}")

    # Summary
    log.info("\n=== Summary ===")
    for beh, code_id in selected_codes.items():
        s = code_stats[code_id]
        rewards = save_dict[f"rewards_{beh}"]
        log.info(
            f"  {beh} (code {code_id}): "
            f"mean_reward={rewards.mean():.1f}, "
            f"ref z_rise={s['z_rise_mean']:.4f}, "
            f"ref xy_disp={s['xy_disp_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
