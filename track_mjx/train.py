"""
Entries point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os

# set default env variable if not set
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"
)
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True "
)

from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import functools
import jax
import wandb
from brax import envs
import orbax.checkpoint as ocp
from track_mjx.agent import custom_ppo
import warnings
from jax import numpy as jp

from datetime import datetime
from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.io import load
from track_mjx.environment import custom_wrappers
from track_mjx.agent import custom_ppo_networks
from track_mjx.agent.logging import rollout_logging_fn, make_rollout_renderer
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.mouse_arm import MouseArm
import logging
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.task.reward import RewardConfig

FLAGS = flags.FLAGS
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="config", config_name="mouse-arm")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("mouse_arm_multi_clip", MultiClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Generates a completely random UUID (version 4), take the first 8 characters
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    model_path = f"./{cfg.logging_config.model_path}/{cfg.env_config.walker_name}_{cfg.data_path}_{run_id}"
    model_path = hydra.utils.to_absolute_path(model_path)
    logging.info(f"Model Checkpoint Path: {model_path}")

    # initialize orbax checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=cfg.train_setup["checkpoint_max_to_keep"],
        keep_period=cfg.train_setup["checkpoint_keep_period"],
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(model_path, options=mgr_options)

    # try to restore configs if previously trained
    if cfg.train_setup["checkpoint_to_restore"] is not None:
        with ocp.CheckpointManager(
            cfg.train_setup["checkpoint_to_restore"],
            options=mgr_options,
        ) as mngr:
            print(f"latest checkpoint step: {mngr.latest_step()}")
            try:
                latest_step = mngr.latest_step()
                abstract_config = OmegaConf.to_container(cfg, resolve=True)
                restored_config = mngr.restore(
                    latest_step,
                    args=ocp.args.Composite(
                        config=ocp.args.JsonRestore(abstract_config)
                    ),
                )["config"]
                print(f"Successfully restored config")
                cfg = OmegaConf.create(restored_config)
            except Exception as e:  # TODO: too broad exception, fix later
                print(
                    f"Failed to restore metadata. Falling back to default cfg: {e}, using current configs"
                )

    env_args = cfg.env_config["env_args"]
    env_rewards = cfg.env_config["reward_weights"]
    train_config = cfg.train_setup["train_config"]
    walker_config = cfg["walker_config"]
    traj_config = cfg["reference_config"]

    # TODO: Fix this dependency issue
    import sys

    logging.info(f"Loading data: {cfg.data_path}")
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    try:
        reference_clip = load.make_multiclip_data(data_path)
    except KeyError:
        logging.info(
            f"Loading from stac-mjx format failed. Loading from ReferenceClip format."
        )
        reference_clip = load.load_reference_clip_data(data_path)

    walker_map = {
        "mouse-arm": MouseArm,
        "rodent": Rodent,
        "fly": Fly,
    }

    walker_class = walker_map[cfg_dict["walker_type"]]
    walker = walker_class(**walker_config)

    reward_config = RewardConfig(**env_rewards)

    # Automatically match dict keys and func needs
    env = envs.get_environment(
        env_name=cfg.env_config.env_name,
        reference_clip=reference_clip,
        walker=walker,
        reward_config=reward_config,
        **env_args,
        **traj_config,
    )

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    episode_length = (
        traj_config.clip_length
        - traj_config.random_init_range
        - traj_config.traj_length
    ) * env._steps_for_cur_frame
    print(f"episode_length {episode_length}")
    logging.info(f"episode_length {episode_length}")

    train_fn = functools.partial(
        custom_ppo.train,
        **train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        kl_weight=cfg.network_config.kl_weight,
        network_factory=functools.partial(
            custom_ppo_networks.make_intention_ppo_networks,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        ),
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
    )

    wandb.init(
        project=cfg.logging_config.project_name,
        config=OmegaConf.to_container(cfg, resolve=True, structured_config_mode=True),
        notes=f"clip_id: {cfg.logging_config.clip_id}",
    )
    wandb.run.name = f"{cfg.env_config.env_name}_{cfg.env_config.task_name}_{cfg.logging_config.algo_name}_{run_id}"

    def wandb_progress(num_steps, metrics):
        metrics["num_steps_thousands"] = num_steps
        wandb.log(metrics, commit=False)

    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)

    # define the jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    renderer, mj_model, mj_data, scene_option = make_rollout_renderer(rollout_env, cfg)
    policy_params_fn = functools.partial(
        rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        model_path,
        renderer,
        mj_model,
        mj_data,
        scene_option,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,
    )


if __name__ == "__main__":
    main()
