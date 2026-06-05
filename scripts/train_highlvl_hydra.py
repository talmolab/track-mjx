"""Unified high-level decoder transfer training script.

Trains a high-level policy that outputs latent intentions to a frozen pretrained
mimic decoder. Supports any task environment registered in vnl-playground.

Usage:
    # Basic usage (any registered task)
    python train_highlvl.py --task RodentBowlEscape --mimic_checkpoint 260210_013247_285744
    python train_highlvl.py --task RodentRearing --mimic_checkpoint 260210_013247_285744
    python train_highlvl.py --task MyCustomTask --mimic_checkpoint 260210_013247_285744

    # With PPO overrides
    python train_highlvl.py --task RodentRearing \\
        --mimic_checkpoint 260210_013247_285744 \\
        --num_timesteps 1e8 --entropy_cost 0.1

    # With env config overrides (dot notation for nested)
    python train_highlvl.py --task RodentBowlEscape \\
        --mimic_checkpoint 260210_013247_285744 \\
        --env "target_speed=1.5 ctrl_dt=0.02"

    # With custom observation keys for policy/value networks
    python train_highlvl.py --task RodentBowlEscape \\
        --mimic_checkpoint 260210_013247_285744 \\
        --policy_obs_key state --value_obs_key privileged_state

    # Warp backend (full-collision rodent model)
    python train_highlvl.py --task RodentBowlEscape \\
        --mimic_checkpoint 260131_223134_344901 \\
        --env "mujoco_impl=warp" --num_envs 1024
"""

import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
from datetime import datetime
from typing import Any

import hydra
import jax
import wandb
from omegaconf import DictConfig
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from mujoco_playground import wrapper
from omegaconf import OmegaConf

from vnl_playground import registry
from vnl_playground.tasks import wrappers as rodent_wrappers
from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from utils import (
    apply_env_overrides,
    configure_warp_backend,
    create_base_parser,
    create_policy_params_fn,
    create_ppo_params,
    get_training_params,
    make_logging_inference_fn,
    parse_env_overrides_str,
    setup_jax_cache,
)

setup_jax_cache()


def load_mimic_checkpoint(checkpoint_path: str) -> tuple:
    """Load mimic checkpoint config and decoder policy."""
    if os.path.isabs(checkpoint_path):
        full_path = checkpoint_path
    else:
        full_path = hydra.utils.to_absolute_path(
            f"./model_checkpoints/{checkpoint_path}"
        )

    mimic_cfg = OmegaConf.create(checkpointing.load_config_from_checkpoint(full_path))
    decoder_policy_fn = ff_ppo_networks.make_decoder_policy_fn(full_path)
    return mimic_cfg, decoder_policy_fn


def create_env_config(task_name: str, mimic_cfg: Any):
    """Create environment config inheriting ctrl_dt from mimic."""
    env_cfg = registry.get_default_config(task_name)
    env_cfg.ctrl_dt = mimic_cfg.env_config.ctrl_dt
    return env_cfg


def create_environments(
    task_name: str,
    env_cfg: Any,
    decoder_policy_fn,
    intention_size: int,
    highlvl_obs_key: str,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
):
    """Create training and eval environments with HighLevelWrapper."""
    wrapper_kwargs = dict(
        decoder_inference_fn=decoder_policy_fn,
        latent_size=intention_size,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
        highlvl_obs_key=highlvl_obs_key,
        lowlvl_obs_key="proprioception",
    )
    env = rodent_wrappers.HighLevelWrapper(
        registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False),
        **wrapper_kwargs,
    )
    eval_env = rodent_wrappers.HighLevelWrapper(
        registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False),
        **wrapper_kwargs,
    )
    return env, eval_env


@hydra.main(
    version_base=None,
    config_path="../track_mjx/config",
    config_name="highlvl",
)
def main(cfg: DictConfig):
    """Main training entry point."""
    print(f"Task: {cfg.task}")

    # Load mimic checkpoint
    print(f"Loading mimic checkpoint: {cfg.mimic_checkpoint}")
    mimic_cfg, decoder_policy_fn = load_mimic_checkpoint(cfg.mimic_checkpoint)

    # Create configs
    env_cfg = create_env_config(cfg.task, mimic_cfg)
    if cfg.env_overrides is not None:
        env_overrides = cfg.env_overrides
    else:
        env_overrides = {}
    ppo_params = cfg.ppo
    # Detect Warp from parsed overrides (before applying to env_cfg)
    default_warp = env_cfg.get("mujoco_impl") == "warp"
    override_warp = env_overrides.get("mujoco_impl") == "warp"
    is_warp = default_warp or override_warp

    if is_warp:
        print("Using warp backend! You may need to lower the number of environments.")

    # Set Warp defaults first, then apply all user overrides (can override naconmax, njmax)
    if is_warp:
        configure_warp_backend(env_cfg, cfg.ppo.num_envs, task_name=cfg.task)
    apply_env_overrides(env_cfg, env_overrides)

    print(f"env_cfg:\n{env_cfg}")
    print(f"ppo_params:\n{ppo_params}")

    # Setup experiment (include microseconds to avoid run ID collisions)
    cfg.logging.id = f"{cfg.task}-highlvl-{cfg.logging.id}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    config_to_save = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=True
    )
    config_to_save["env_config"] = env_cfg.to_dict()
    ckpt_path = epath.Path(cfg.checkpoint_dir).resolve() / cfg.logging.id
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment name: {cfg.logging.id}")
    print(f"Checkpoint path: {ckpt_path}")

    # Initialize wandb
    wandb.init(
        **cfg.logging,
        config=config_to_save,
        notes=f"task: {cfg.task}, mimic: {cfg.mimic_checkpoint}",
    )

    def wandb_progress(num_steps, metrics):
        print(f"Step {num_steps}")
        print(f"Metrics: {metrics}")
        wandb.log(metrics)

    # Create environments
    env, eval_env = create_environments(
        cfg.task,
        env_cfg,
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        cfg.obs_keys.highlvl,
        policy_obs_key=cfg.obs_keys.policy,
        value_obs_key=cfg.obs_keys.value,
    )

    try:
        wandb.log(
            {"spec_file": wandb.Html(env.save_spec("./env_spec.xml", return_str=True))},
            commit=False,
        )
    except Exception as e:
        print(f"Error saving spec: {e}")

    # Setup training
    training_params = get_training_params(ppo_params)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_obs_key=cfg.obs_keys.policy,
        value_obs_key=cfg.obs_keys.value,
        **ppo_params.network_factory,
    )
    normalize = lambda x, _y: x
    if training_params["normalize_observations"]:
        print("Normalizing observations")
        normalize = running_statistics.normalize

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        seed=cfg.seed,
        num_evals=int(ppo_params.num_timesteps / ppo_params.eval_every),
        network_factory=network_factory,
        restore_checkpoint_path=None,
        progress_fn=wandb_progress,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training),
    )

    # Setup logging
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(cfg.seed)
    start_state = jit_reset(rng)

    # Get observation size as nested dict (Brax handles per-key extraction)
    obs_size = jax.tree.map(lambda x: x.shape[-1], start_state.obs)

    ppo_network = network_factory(
        obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    policy_params_fn = create_policy_params_fn(
        ppo_params,
        ckpt_path,
        env,
        jit_reset,
        jit_step,
        jit_logging_inference_fn,
        rollout_metrics=cfg.get("rollout_metrics", None),
    )

    # Run training
    train_fn(environment=env, eval_env=eval_env, policy_params_fn=policy_params_fn)
    print("Training complete!")


if __name__ == "__main__":
    main()
