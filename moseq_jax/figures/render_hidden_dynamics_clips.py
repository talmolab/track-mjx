"""Render verification videos for hidden dynamics single-code clips.

Re-runs the walk and groom clips from run_hidden_dynamics, collects qpos,
and renders one ghost video per behavior to verify clip identity.

Usage:
    cd moseq_jax/figures
    python render_hidden_dynamics_clips.py
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
# Prevent PyTorch's CUDA libs from conflicting with JAX's cuDNN
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

import sys
from collections import OrderedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MOSEQ_DIR = SCRIPT_DIR.parent
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
    run_rollout,
)
from experiments.shared.clip_selection import (
    load_balanced_splits,
    select_clips_by_behavior,
)
from experiments.shared.code_sequences import make_correct_sequences
from experiments.shared.ghost_rendering import (
    build_ghost_model,
    render_ghost_video,
)
from experiments.shared.plotting import get_trajectory_colors

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "hidden_dynamics"

# Same config as run_hidden_dynamics
CKPT_PATH = "/home/jovyan/vast/kaiwen/track-mjx/moseq_jax/model_checkpoints/260407_031233_484020/"
CODES_PATH = "/home/jovyan/vast/kaiwen/track-mjx/moseq_jax/outputs/kpms_sweep/best_codes.npz"
SPLIT_PATH = "/home/jovyan/vast/kaiwen/track-mjx/data/rodent/rodent_balanced_splits.json"
REF_PATH = "/home/jovyan/vast/kaiwen/track-mjx/data/rodent/rodent_reference_clips.h5"

K = 10
SEED = 42
MAX_STEPS = 250


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(CKPT_PATH)
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    params = (norm_state, policy_params)

    # Load codes and splits
    codes_data = np.load(CODES_PATH)
    test_codes = codes_data["test_codes"]
    splits = load_balanced_splits(SPLIT_PATH)
    test_indices = splits["balanced"]["test_indices"]

    # Build env
    test_clips = ReferenceClips(
        data_path=REF_PATH,
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)

    # Select same clips as hidden dynamics experiment
    selected = select_clips_by_behavior(splits, "test", k_per_behavior=K, seed=SEED)

    for beh in ["walk", "groom"]:
        beh_indices = selected.get(beh, [])[:K]
        if not beh_indices:
            continue

        code_seqs = make_correct_sequences(test_codes, beh_indices, MAX_STEPS)

        print(f"\n=== {beh.upper()}: running {len(beh_indices)} clips ===")
        all_qpos = []
        all_codes = []

        for ki, ci in enumerate(beh_indices):
            key = jax.random.PRNGKey(SEED + ki * 1000)
            result = run_rollout(
                env, inf_fn, params, ppo_networks, use_rnn, key,
                max_steps=MAX_STEPS,
                code_override=code_seqs[ki],
                reset_clip_idx=ci,
                jit_reset=jit_reset, jit_step=jit_step,
                ignore_done=True,
            )
            qpos = result["qpos"][:-1]  # trim extra final frame
            codes = result["code_indices"]
            n_unique = len(np.unique(code_seqs[ki]))
            print(f"  clip {ki}: {len(qpos)} frames, {n_unique} unique codes, survival={result['survival']}")
            all_qpos.append(qpos)
            all_codes.append(codes)

        # Render ghost video with all clips
        min_len = min(len(q) for q in all_qpos)
        trimmed_qpos = [q[:min_len] for q in all_qpos]
        trimmed_codes = [c[:min_len] for c in all_codes]

        n_bodies = len(trimmed_qpos)
        traj_colors = get_trajectory_colors(n_bodies)

        print(f"  Building ghost model ({n_bodies} bodies)...")
        ghost_model, base_nq = build_ghost_model(
            env,
            num_ghosts=n_bodies - 1,
            ghost_colors=traj_colors[1:],
            camera_distance=0.8,
            camera_elevation=-30.0,
            camera_azimuth=135.0,
        )

        video_path = OUTPUT_DIR / f"verify_{beh}_clips.mp4"
        print(f"  Rendering {video_path}...")
        render_ghost_video(
            ghost_model=ghost_model,
            base_nq=base_nq,
            trajectories_qpos=trimmed_qpos,
            code_sequences=trimmed_codes,
            trajectory_colors=traj_colors,
            output_path=video_path,
            title=f"{beh} clips (n={n_bodies})",
            width=800,
            height=600,
        )
        print(f"  Saved: {video_path}")


if __name__ == "__main__":
    main()
