"""Experiment: RNN Hidden State Dynamics.

Records GRU hidden states at each timestep for walk/groom/rear code
sequences to visualize behavior-specific dynamical regimes in the
recurrent latent space.

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
from experiments.shared.clip_selection import (
    load_balanced_splits,
    select_clips_by_behavior,
)
from experiments.shared.code_sequences import make_correct_sequences

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom rollout that captures hidden states
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
    """Run a single rollout collecting GRU hidden states at every step.

    The rollout never breaks on ``state.done`` so that all trajectories
    have uniform length ``max_steps`` (required for stacking into a
    regular array downstream).

    Args:
        env: ``MoSeqImitation`` environment.
        inference_fn: JIT-compiled RNN inference function.
        params: ``(normalizer_state, policy_params)`` tuple.
        ppo_networks: Network object (for ``init_hidden``).
        key: PRNG key.
        max_steps: Rollout length.
        code_override: ``[max_steps]`` int array to override codes.
        reset_clip_idx: Clip index to reset to.
        jit_reset: Pre-compiled ``jax.jit(env.reset)``.
        jit_step: Pre-compiled ``jax.jit(env.step)``.

    Returns:
        Dict with ``hidden_states`` ``[T, hidden_dim]``,
        ``code_indices`` ``[T]``, ``rewards`` ``[T]``,
        ``survival`` int.
    """
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
    reward_list: list[float] = []
    survival = max_steps  # updated on first done

    for t in range(max_steps):
        # Override code observation
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

        # Record hidden state (last GRU layer, unbatched)
        hidden_list.append(np.array(hidden[-1][0]))

        # Record code index
        if "code_idx" in extras:
            code_list.append(int(extras["code_idx"]))
        elif "indices" in extras:
            code_list.append(int(extras["indices"]))

        # Step environment
        state = jit_step(state, action)
        reward_list.append(float(state.reward))

        if state.done and survival == max_steps:
            survival = t + 1

    return {
        "hidden_states": np.array(hidden_list),   # [T, hidden_dim]
        "code_indices": np.array(code_list),       # [T]
        "rewards": np.array(reward_list),          # [T]
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
    # Load codes and splits
    # ------------------------------------------------------------------
    codes_data = np.load(cfg.data.codes_path)
    test_codes = codes_data["test_codes"]
    splits = load_balanced_splits(cfg.data.balanced_split_path)
    test_indices = splits["balanced"]["test_indices"]

    # ------------------------------------------------------------------
    # Build environment (same pattern as other experiments)
    # ------------------------------------------------------------------
    test_clips = ReferenceClips(
        data_path=cfg.data.reference_data_path,
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
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

    # ------------------------------------------------------------------
    # Select clips and run rollouts
    # ------------------------------------------------------------------
    K = int(cfg.K)
    seed = int(cfg.seed)
    max_steps = int(cfg.max_steps)
    behaviors = ["walk", "groom", "rear"]

    selected = select_clips_by_behavior(
        splits, "test", k_per_behavior=K, seed=seed,
    )

    save_dict: dict[str, np.ndarray] = {}

    for beh in behaviors:
        beh_indices = selected.get(beh, [])[:K]
        if not beh_indices:
            log.warning(f"No clips for {beh}, skipping")
            continue

        code_seqs = make_correct_sequences(test_codes, beh_indices, max_steps)

        beh_hidden: list[np.ndarray] = []
        beh_codes: list[np.ndarray] = []
        beh_survivals: list[int] = []

        log.info(
            f"Running {len(beh_indices)} {beh} clips "
            f"(max_steps={max_steps})..."
        )

        for ki, ci in enumerate(beh_indices):
            key = jax.random.PRNGKey(seed + ki * 1000)
            result = run_rollout_with_hidden(
                env,
                inf_fn,
                params,
                ppo_networks,
                key,
                max_steps=max_steps,
                code_override=code_seqs[ki],
                reset_clip_idx=ci,
                jit_reset=jit_reset,
                jit_step=jit_step,
            )
            beh_hidden.append(result["hidden_states"])
            beh_codes.append(result["code_indices"])
            beh_survivals.append(result["survival"])
            log.info(
                f"  {beh} clip {ki}: "
                f"survival={result['survival']}/{max_steps}"
            )

        # Stack into regular arrays [K, T, dim]
        save_dict[f"hidden_{beh}"] = np.stack(beh_hidden)
        save_dict[f"codes_{beh}"] = np.stack(beh_codes)
        save_dict[f"survivals_{beh}"] = np.array(beh_survivals)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = data_dir / "hidden_dynamics.npz"
    np.savez_compressed(out_path, **save_dict)
    log.info(f"Saved: {out_path}")

    # Copy to figures/data for the plotting script
    fig_data_dir = MOSEQ_DIR / "figures" / "data"
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_data_dir / "hidden_dynamics.npz"
    shutil.copy2(out_path, dst)
    log.info(f"Copied to: {dst}")

    # Summary
    for beh in behaviors:
        key = f"survivals_{beh}"
        if key in save_dict:
            s = save_dict[key]
            log.info(
                f"  {beh}: {len(s)} clips, "
                f"mean survival={np.mean(s):.0f}/{max_steps}"
            )


if __name__ == "__main__":
    main()
