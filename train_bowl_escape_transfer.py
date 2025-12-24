"""
Entry point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os
import sys

# set default env variable if not set
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
#     "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6"
# )

# # limit to 1 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # visible GPU masks

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import jax

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import hydra
from omegaconf import DictConfig, OmegaConf
import functools
import wandb
import orbax.checkpoint as ocp
from track_mjx.agent.mlp_ppo import ppo, ppo_networks
import warnings
from pathlib import Path
from datetime import datetime
import logging
import mujoco
from mujoco_playground import wrapper
from vnl_playground.tasks.rodent import bowl_escape
from vnl_playground.tasks.rodent import wrappers as rodent_wrappers


from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="./", config_name="bowl_escape_transfer")
def main(bowl_cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    # logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")

    # Generate a new run_id and associated checkpoint path
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_path = hydra.utils.to_absolute_path(f"./model_checkpoints/{run_id}")

    # Load the checkpoint's config
    # TODO: We set the restored config's checkpoint_to_restore to itself
    # Because that restored config is used from now on. This is a hack.
    checkpoint_to_restore = hydra.utils.to_absolute_path(
        "./model_checkpoints/251223_232558_038379"
    )
    # Load the checkpoint's config and update the run_id and checkpoint path
    loaded_cfg = OmegaConf.create(
        checkpointing.load_config_from_checkpoint(checkpoint_to_restore)
    )

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )

    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    logging.info(f"run_id: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")
    print(loaded_cfg)
    ppo_params = {
        "episode_length": 2000,
        "num_envs": 4096,
        "num_timesteps": 500_000_000,
        "batch_size": 1024,
        "num_minibatches": 16,
        "num_updates_per_batch": 4,
        "learning_rate": 1e-4,
        "clipping_epsilon": 0.1,
        "discounting": 0.98,
        "action_repeat": 1,
        "entropy_cost": 1e-4,
        "reward_scaling": 1,
        "normalize_observations": True,
        "unroll_length": 20,
        "seed": 0,
    }

    env_config = {
        "torque_actuators": True,
        "rescale_factor": 0.9,
        "bowl_vsize": 0.6,
        "bowl_amplitude": -20,
        "target_speed": 2,
        "iterations": 10,
        "ctrl_dt": loaded_cfg.env_config.ctrl_dt,
    }

    env = rodent_wrappers.FlattenObsWrapper(
        bowl_escape.BowlEscape(config_overrides=env_config)
    )
    evaluator_env = rodent_wrappers.FlattenObsWrapper(
        bowl_escape.BowlEscape(config_overrides=env_config)
    )

    train_fn = functools.partial(
        ppo.train,
        **ppo_params,
        num_evals=int(ppo_params["num_timesteps"] / 5_000_000),
        num_resets_per_eval=1,
        latent_kl_weight=0,
        latent_ar1_weight=0,
        network_factory=functools.partial(
            ppo_networks.make_intention_ppo_networks,
            encoder_hidden_layer_sizes=tuple(
                loaded_cfg.network_config.encoder_layer_sizes
            ),
            decoder_hidden_layer_sizes=tuple(
                loaded_cfg.network_config.decoder_layer_sizes
            ),
            value_hidden_layer_sizes=tuple(
                loaded_cfg.network_config.critic_layer_sizes
            ),
            intention_latent_size=loaded_cfg.network_config.intention_size,
        ),
        latent_kl_weight=bowl_cfg.network_config.latent_kl_weight,
        latent_ar1_weight=bowl_cfg.network_config.latent_ar1_weight,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=checkpoint_to_restore,
        freeze_decoder=True,
        config_dict=OmegaConf.to_container(
            bowl_cfg, resolve=True
        ),  # finalize config here
        use_kl_schedule=False,
        wrap_for_training=functools.partial(
            wrapper.wrap_for_brax_training, full_reset=False
        ),
    )

    run_id = f"{bowl_cfg.logging_config.exp_name}_rodent_bowl_escape_{run_id}"
    wandb.init(
        project=bowl_cfg.logging_config.project_name,
        config=OmegaConf.to_container(
            bowl_cfg, resolve=True, structured_config_mode=True
        ),
        notes=f"{bowl_cfg.logging_config.notes}",
        id=run_id,
        resume="allow",
        group=bowl_cfg.logging_config.group_name,
    )

    def wandb_progress(num_steps, metrics):
        metrics["num_steps_thousands"] = num_steps
        wandb.log(metrics, commit=False)

    # # define the jit reset/step functions
    jit_reset = jax.jit(evaluator_env.reset)
    jit_step = jax.jit(evaluator_env.step)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        env=evaluator_env,
        jit_reset=jit_reset,
        jit_step=jit_step,
        cfg=loaded_cfg,
        model_path=checkpoint_path,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,
    )


if __name__ == "__main__":
    main()
