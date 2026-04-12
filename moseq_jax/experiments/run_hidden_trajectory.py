"""Experiment: Natural Hidden State Trajectories.

Samples long continuous segments from unseen data, runs KPMS to extract
syllable codes, then feeds the natural code sequences through the trained
Code2Act decoder to collect hidden states at every timestep.

The resulting hidden state trajectories show how the RNN's internal state
flows through latent space during natural behavior, with each timestep
labeled by its active KPMS code.

Usage:
    cd moseq_jax
    python -m experiments.run_hidden_trajectory
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
from experiments.shared.keypoint_fk import setup_stac_model, qpos_to_keypoints_fk

# Re-use generalization helpers for sampling + KPMS
from experiments.run_generalization import (
    sample_segments,
    write_segmented_h5,
    extract_keypoints,
    run_kpms_inference,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom rollout collecting hidden states
# ---------------------------------------------------------------------------


def run_rollout_with_hidden(
    env,
    inference_fn,
    params,
    ppo_networks,
    key,
    max_steps: int,
    code_override: np.ndarray,
    reset_clip_idx: int,
    jit_reset=None,
    jit_step=None,
) -> dict[str, np.ndarray]:
    """Run a single rollout collecting GRU hidden states and qpos."""
    if jit_reset is None:
        jit_reset = jax.jit(env.reset)
    if jit_step is None:
        jit_step = jax.jit(env.step)

    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key, clip_idx=jnp.int32(reset_clip_idx))

    hidden = ppo_networks.policy_network.init_hidden(1)
    _code_stack_size = int(state.obs["kpms_code"].shape[-1])

    hidden_list: list[np.ndarray] = []
    code_list: list[int] = []
    qpos_list: list[np.ndarray] = []
    reward_list: list[float] = []
    survival = max_steps

    for t in range(max_steps):
        # Override code
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
    config_name="hidden_trajectory_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Natural Hidden State Trajectory Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    K = int(cfg.K)
    frames_per_segment = int(cfg.frames_per_segment)
    seed = int(cfg.seed)

    # ------------------------------------------------------------------
    # Step 1: Sample long segments from continuous data
    # ------------------------------------------------------------------
    log.info(f"Sampling {K} segments of {frames_per_segment} frames...")
    seg_data = sample_segments(
        cfg.new_data.path, n_segments=K,
        frames_per_segment=frames_per_segment, seed=seed,
    )

    seg_h5_path = data_dir / "segments.h5"
    write_segmented_h5(seg_data, str(seg_h5_path))

    # ------------------------------------------------------------------
    # Step 2: FK → keypoints
    # ------------------------------------------------------------------
    log.info("Extracting keypoints via FK...")
    kps, kp_names = extract_keypoints(
        seg_data["qpos"], cfg.reference_h5, cfg.stac_xml,
    )

    # ------------------------------------------------------------------
    # Step 3: KPMS → codes
    # ------------------------------------------------------------------
    log.info("Running KPMS inference...")
    kps = kps.astype(np.float64)  # KPMS requires float64 under jax_enable_x64
    codes = run_kpms_inference(
        kps, n_segments=K, frames_per_segment=frames_per_segment,
        model_dir=cfg.kpms.model_dir, model_name=cfg.kpms.model_name,
        num_iters=int(cfg.kpms.num_iters),
    )
    log.info(f"  Codes shape: {codes.shape}")

    # Save codes
    np.savez_compressed(
        data_dir / "trajectory_codes.npz",
        codes=codes,
        start_indices=seg_data["start_indices"],
    )

    # ------------------------------------------------------------------
    # Step 4: Load checkpoint and build env
    # ------------------------------------------------------------------
    log.info("Loading checkpoint...")
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path,
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    if not use_rnn:
        raise ValueError("This experiment requires an RNN decoder checkpoint.")
    params = (norm_state, policy_params)

    # Build env with the segmented clips
    seg_clips = ReferenceClips(
        data_path=str(seg_h5_path),
        n_frames_per_clip=frames_per_segment,
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    env_cfg.clip_length = frames_per_segment
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg,
        clips=seg_clips,
        kpms_codes=codes,
        code_stack_size=code_stack_size,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inf_fn = make_inference_fn(ppo_networks, use_rnn=True, deterministic=True)

    # ------------------------------------------------------------------
    # Step 5: Run rollouts collecting hidden states
    # ------------------------------------------------------------------
    log.info(f"Running {K} rollouts ({frames_per_segment} frames each)...")

    all_hidden: list[np.ndarray] = []
    all_codes: list[np.ndarray] = []
    all_qpos: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_survivals: list[int] = []

    for ki in range(K):
        key = jax.random.PRNGKey(seed + ki * 1000)
        result = run_rollout_with_hidden(
            env, inf_fn, params, ppo_networks, key,
            max_steps=frames_per_segment,
            code_override=codes[ki],
            reset_clip_idx=ki,
            jit_reset=jit_reset, jit_step=jit_step,
        )
        all_hidden.append(result["hidden_states"])
        all_codes.append(result["code_indices"])
        all_qpos.append(result["qpos"])
        all_rewards.append(result["rewards"])
        all_survivals.append(result["survival"])

        n_unique = len(np.unique(codes[ki]))
        log.info(
            f"  segment {ki}: survival={result['survival']}/{frames_per_segment}, "
            f"{n_unique} unique codes, mean_reward={result['rewards'].mean():.1f}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = data_dir / "hidden_trajectory.npz"
    np.savez_compressed(
        out_path,
        hidden=np.stack(all_hidden),       # [K, T, 256]
        codes=np.stack(all_codes),         # [K, T]
        qpos=np.stack(all_qpos),           # [K, T, 74]
        rewards=np.stack(all_rewards),     # [K, T]
        survivals=np.array(all_survivals), # [K]
        kpms_codes=codes,                  # [K, T] original KPMS codes
    )
    log.info(f"Saved: {out_path}")

    # Copy to figures/data
    fig_data_dir = MOSEQ_DIR / "figures" / "data"
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_data_dir / "hidden_trajectory.npz"
    shutil.copy2(out_path, dst)
    log.info(f"Copied to: {dst}")

    # Summary
    log.info("\n=== Summary ===")
    all_codes_flat = np.concatenate(all_codes)
    unique, counts = np.unique(all_codes_flat, return_counts=True)
    order = np.argsort(-counts)
    log.info(f"  Total frames: {len(all_codes_flat)}")
    log.info(f"  Active codes: {len(unique)}")
    log.info(f"  Top 10 codes: {[(int(unique[o]), int(counts[o])) for o in order[:10]]}")
    log.info(f"  Mean survival: {np.mean(all_survivals):.0f}/{frames_per_segment}")


if __name__ == "__main__":
    main()
