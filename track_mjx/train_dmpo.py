"""Track-mjx DMPO training entry — imitation only.

For DMPO on downstream tasks (gap, vision, walker, etc.) use
``vnl_playground.train_dmpo`` in the vnl-playground repo. Track-mjx hosts
only the imitation entry; downstream tasks live with their env definitions.

Usage:
    python -m track_mjx.train_dmpo --config-name=rodent-dmpo-imitation
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from vnl_playground import registry as vp_registry

from track_mjx.agent.dmpo.checkpoint import (
    make_checkpointer,
    restore as restore_ckpt,
    save as save_ckpt,
)
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state, make_optimizers
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.train import (
    _VnlPlaygroundEnvAdapter,
    _filter_dmpo_kwargs,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    detect_git_sha,
    load_wandb_state as load_dmpo_wandb_state,
    make_run_id,
    save_wandb_state as save_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.training_loop import run as run_training_loop
from track_mjx.config import utils as cfg_utils

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_IMPORTED = True
except ImportError:
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


@hydra.main(
    config_path="config", config_name="rodent-dmpo-imitation", version_base=None
)
def main(hydra_cfg: DictConfig):
    """DMPO training entry — rodent imitation only."""
    # Resolve walker paths into env_config and produce the ConfigDict the
    # registry expects. Mirrors track_mjx/train.py:62 (the PPO entry).
    hydra_cfg, cfg_dict, env_cfg_ml = cfg_utils.prepare_config(hydra_cfg)

    raw_train_cfg = dict(cfg_dict["train_config"])
    cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    iters_per_chunk = int(raw_train_cfg.get("iters_per_chunk", 32))
    log.info("iters_per_chunk=%d", iters_per_chunk)
    seed = int(hydra_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    # ---- 0. Wandb (deterministic name + resume) ----
    config_name = str(
        hydra_cfg.get("logging_config", {}).get(
            "exp_name", hydra_cfg.get("env_name", "dmpo-imitation")
        )
    )
    git_sha = detect_git_sha(Path(__file__).resolve().parents[1])
    run_id = make_run_id(config_name, seed, git_sha)
    log.info("wandb run_id=%s", run_id)

    ckpt_dir = str(hydra_cfg.get("checkpoint_dir", "./checkpoints/dmpo_imitation"))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    existing = load_dmpo_wandb_state(ckpt_dir)
    if _WANDB_IMPORTED:
        try:
            wandb.init(
                project=str(
                    hydra_cfg.get("logging_config", {}).get(
                        "project_name", "dmpo-rodent-imitation"
                    )
                ),
                config=cfg_dict,
                mode=os.environ.get("WANDB_MODE", "online"),
                id=existing["wandb_run_id"] if existing else run_id,
                name=existing["wandb_run_id"] if existing else run_id,
                resume="must" if existing else "allow",
                group=str(
                    hydra_cfg.get("logging_config", {}).get(
                        "group_name", hydra_cfg.get("env_name", "dmpo-imitation")
                    )
                ),
                notes=str(hydra_cfg.get("logging_config", {}).get("notes", "")),
                reinit=True,
            )
            save_dmpo_wandb_state(
                ckpt_dir,
                run_id if not existing else existing["wandb_run_id"],
            )
        except Exception as e:
            log.warning("wandb.init failed (%s); continuing without wandb.", e)

    # ---- 1. Load env (imitation: clips + registry.load, flat obs) ----
    env_name = str(hydra_cfg.env_config.env_name)
    log.info("Loading imitation env via vnl_playground.registry: %s", env_name)
    reference_clips = vp_registry.load_reference_clips(
        env_name,
        data_path=hydra_cfg.env_config.reference_data_path,
        n_frames_per_clip=hydra_cfg.env_config.clip_length,
        keep_clips_idx=hydra_cfg.env_config.get("keep_clips_idx", None),
    )
    raw_env = vp_registry.load(
        env_name, config=env_cfg_ml, clips=reference_clips, flatten_obs=True,
    )
    env = _VnlPlaygroundEnvAdapter(raw_env)
    obs_size = int(env.observation_size)
    action_size = int(env.action_size)
    env_spec = {"obs_size": obs_size, "action_size": action_size}
    log.info("env_spec: obs_size=%d action_size=%d", obs_size, action_size)

    # ---- 2. Networks + training state + optimizers (flat-obs path) ----
    nets = make_dmpo_networks(obs_size, action_size, cfg)
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    # ---- 2a. Checkpoint manager + restore-on-resume ----
    ckpt_mgr = make_checkpointer(ckpt_dir)
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    if restored is not None:
        log.info(
            "Restored DMPO checkpoint at training step %d", int(restored.steps)
        )
        state = restored

    # ---- 3. Replay (flashbax) ----
    transition_template = {
        "observation": jnp.zeros((obs_size,), dtype=jnp.float32),
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": jnp.zeros((obs_size,), dtype=jnp.float32),
    }
    rb = make_replay(
        max_size=max(
            cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs
        ),
        min_size=max(
            cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs
        ),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    # ---- 4. K (SGD updates per fused_step) ----
    K = max(
        1,
        int(
            cfg.unroll_length * cfg.num_envs
            / (cfg.batch_size * cfg.samples_per_insert)
        ),
    )
    log.info("DMPO imitation: K=%d SGD updates per rollout", K)

    # ---- 5. Eval / wandb / ckpt callbacks (no vision render) ----
    def wandb_log_cb(payload: dict, env_steps: int) -> None:
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(payload, step=int(env_steps))

    def ckpt_save_cb(state: object, env_steps: int) -> None:
        save_ckpt(ckpt_mgr, int(env_steps), state, config=cfg_dict)

    # No eval rollout for imitation (Phase 1) — adding it later requires
    # plugging in the imitation eval renderer, which is task-specific.

    # ---- 6. Run loop ----
    rng, k_run = jax.random.split(rng)
    state, env_state, rb_state, last_metrics = run_training_loop(
        env=env,
        nets=nets,
        optimizers=optimizers,
        rb=rb,
        cfg=cfg,
        K=K,
        iters_per_chunk=iters_per_chunk,
        rng=k_run,
        state=state,
        env_state=None,
        rb_state=rb_state,
        wandb_log_callback=wandb_log_cb,
        ckpt_mgr=ckpt_mgr,
        ckpt_save_callback=ckpt_save_cb,
        cfg_dict=cfg_dict,
    )

    ckpt_mgr.wait_until_finished()
    log.info(
        "Training complete: final policy_loss=%.4g critic_loss=%.4g",
        float(last_metrics.get("policy_loss", 0.0)),
        float(last_metrics.get("critic_loss", 0.0)),
    )
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
