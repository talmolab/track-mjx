"""Step 1 training: Multi-behavior MLP + PPO for PlanarWalker.

Trains a conditional policy pi(a | s, mode_onehot) with 4 behavior modes.
Uses Brax PPO with standard MLP networks (no encoder-decoder).

Usage:
    cd /home/talmolab/Desktop/SalkResearch/track-mjx
    source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
    python track_mjx/train_walker_multi_behavior.py
    # or with overrides:
    python track_mjx/train_walker_multi_behavior.py env_config.fixed_mode=0
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import jax.numpy as jp
import orbax.checkpoint as ocp
from brax.training.agents.ppo import train as brax_ppo_train
from brax.training.agents.ppo import networks as brax_ppo_networks
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf

from vnl_playground.tasks.walker.multi_behavior import (
    MultiBehaviorWalker,
    default_config as walker_default_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="walker-multi-behavior",
)
def main(cfg: DictConfig) -> None:
    """Train multi-behavior walker with Brax PPO."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    try:
        n_devices = jax.device_count(backend="gpu")
        logger.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logger.info("No GPU, using CPU")

    # Create environment config
    env_cfg = walker_default_config()
    for key in [
        "sim_dt", "ctrl_dt", "episode_length", "mujoco_impl",
        "nconmax", "njmax", "mode_switch_prob", "fixed_mode",
    ]:
        if key in cfg.env_config:
            val = cfg.env_config[key]
            if val is not None or key == "fixed_mode":
                env_cfg[key] = val

    # Create train and eval environments
    env = MultiBehaviorWalker(config=env_cfg)
    eval_env = MultiBehaviorWalker(config=env_cfg)

    # Wrap for Brax training
    wrapped_env = playground_wrappers.wrap_for_brax_training(
        env,
        episode_length=cfg.env_config.episode_length,
    )
    wrapped_eval_env = playground_wrappers.wrap_for_brax_training(
        eval_env,
        episode_length=cfg.env_config.episode_length,
    )

    # Setup wandb
    if cfg.logging_config.log_to_wandb:
        import wandb
        wandb.init(
            project=cfg.logging_config.project_name,
            group=cfg.logging_config.group_name,
            name=cfg.logging_config.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def progress_fn(num_steps, metrics):
        logger.info(
            f"Step {num_steps}: "
            f"reward={metrics.get('eval/episode_reward', 0):.3f}"
        )
        if cfg.logging_config.log_to_wandb:
            import wandb
            wandb.log({"step": num_steps, **metrics})

    # Setup checkpoint
    os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
    ckpt_mgr = ocp.CheckpointManager(
        cfg.checkpoint.save_dir,
        options=ocp.CheckpointManagerOptions(create=True),
    )

    # Network factory with configurable hidden sizes
    network_factory = functools.partial(
        brax_ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(
            cfg.network_config.policy_hidden_layer_sizes
        ),
        value_hidden_layer_sizes=tuple(
            cfg.network_config.value_hidden_layer_sizes
        ),
    )

    # Train
    make_inference_fn, params, metrics = brax_ppo_train(
        environment=wrapped_env,
        eval_env=wrapped_eval_env,
        num_timesteps=cfg.train_config.num_timesteps,
        num_envs=cfg.train_config.num_envs,
        batch_size=cfg.train_config.batch_size,
        num_minibatches=cfg.train_config.num_minibatches,
        num_updates_per_batch=cfg.train_config.num_updates_per_batch,
        learning_rate=cfg.train_config.learning_rate,
        entropy_cost=cfg.train_config.entropy_cost,
        discounting=cfg.train_config.discounting,
        unroll_length=cfg.train_config.unroll_length,
        seed=cfg.train_config.seed,
        normalize_observations=cfg.train_config.normalize_observations,
        network_factory=network_factory,
        progress_fn=progress_fn,
        num_evals=int(
            cfg.train_config.num_timesteps / cfg.checkpoint.save_every
        ),
    )

    # Save final checkpoint
    ckpt_mgr.save(
        cfg.train_config.num_timesteps,
        args=ocp.args.StandardSave(params),
    )
    logger.info(f"Final checkpoint saved to {cfg.checkpoint.save_dir}")

    if cfg.logging_config.log_to_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
