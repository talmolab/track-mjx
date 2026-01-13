"""Bowl escape task transfer training with Brax PPO.

Two training modes:
- decoder_only: Freeze decoder, train new encoder
- prior_decoder: Freeze prior + decoder, train residual encoder

Usage:
    python -m track_mjx.agent.task_transfer.bowl_escape.train mode=decoder_only
    python -m track_mjx.agent.task_transfer.bowl_escape.train mode=prior_decoder
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Set XLA flags for JAX
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import jax
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as brax_ppo
from brax.training.acme import running_statistics
from mujoco_playground import wrapper
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import bowl_escape

from track_mjx.agent.task_transfer.bowl_escape.checkpoint_utils import (
    load_prior_checkpoint,
    make_decoder_inference_fn,
    make_prior_inference_fn,
)
from track_mjx.agent.task_transfer.bowl_escape.logging import (
    rollout_logging_fn,
    wandb_progress,
)
from track_mjx.agent.task_transfer.bowl_escape.wrappers import (
    DecoderHighLevelWrapper,
    PriorDecoderHighLevelWrapper,
)

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra config."""
    # Log device info
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except Exception:
        n_devices = 1
        logging.info("Not using GPUs")

    # Generate run ID and paths
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    model_path = hydra.utils.to_absolute_path(
        f"./model_checkpoints/bowl_escape_transfer_{cfg.mode}_{run_id}"
    )
    Path(model_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Model checkpoint path: {model_path}")

    # Load mlp_prior checkpoint
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint.path)
    logging.info(f"Loading checkpoint from: {checkpoint_path}")

    prior_params, decoder_params, normalizer_params, ckpt_cfg = load_prior_checkpoint(
        checkpoint_path, cfg.checkpoint.step
    )

    latent_size = ckpt_cfg["network_config"]["intention_size"]
    proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
    logging.info(f"Latent size: {latent_size}, Proprio size: {proprio_size}")

    # Create inference functions
    decoder_fn = make_decoder_inference_fn(decoder_params, normalizer_params, ckpt_cfg)

    # Create environment config
    env_cfg = bowl_escape.default_config()
    env_cfg.ctrl_dt = ckpt_cfg["env_config"]["ctrl_dt"]
    env_cfg.target_speed = cfg.env.target_speed
    logging.info(f"Environment config: ctrl_dt={env_cfg.ctrl_dt}, target_speed={env_cfg.target_speed}")

    # Create training and eval environments with appropriate wrapper
    # Wrappers handle both dict and flat observations natively
    def make_wrapped_env(is_eval: bool = False):
        base_env = bowl_escape.BowlEscape(config=env_cfg)

        if cfg.mode == "decoder_only":
            return DecoderHighLevelWrapper(
                base_env, decoder_fn, latent_size, proprio_size
            )
        else:  # prior_decoder
            prior_fn = make_prior_inference_fn(
                prior_params, normalizer_params, ckpt_cfg
            )
            return PriorDecoderHighLevelWrapper(
                base_env,
                prior_fn,
                decoder_fn,
                latent_size,
                proprio_size,
                deterministic_prior=cfg.prior_decoder.deterministic_prior,
                noise_logvar=cfg.prior_decoder.noise_logvar,
            )

    env = make_wrapped_env(is_eval=False)
    eval_env = make_wrapped_env(is_eval=True)

    logging.info(f"Training mode: {cfg.mode}")
    logging.info(f"Environment action size (latent): {env.action_size}")

    # Save config to model path
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["checkpoint_config"] = ckpt_cfg
    with open(Path(model_path) / "config.json", "w") as fp:
        json.dump(config_dict, fp, indent=4, default=str)

    # Initialize wandb
    wandb_run_id = f"{cfg.logging.exp_name}_{run_id}"
    wandb.init(
        project=cfg.logging.project,
        group=cfg.logging.group_name,
        config=config_dict,
        id=wandb_run_id,
        notes=f"checkpoint: {cfg.checkpoint.path}, mode: {cfg.mode}",
    )

    # Create network factory for standard MLP PPO
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.network.policy_layers),
        value_hidden_layer_sizes=tuple(cfg.network.value_layers),
    )

    # Setup normalization
    normalize = lambda x, y: x
    if cfg.train.normalize_observations:
        normalize = running_statistics.normalize

    # Create PPO networks for inference function setup
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(cfg.train.seed)
    start_state = jit_reset(rng)

    ppo_network = network_factory(
        start_state.obs.shape[-1],
        eval_env.action_size,
        preprocess_observations_fn=normalize,
    )

    # Create logging inference function
    def make_logging_inference_fn(ppo_networks):
        """Create inference function for logging rollouts."""

        def make_logging_policy(deterministic=True):
            policy_network = ppo_networks.policy_network
            parametric_action_distribution = ppo_networks.parametric_action_distribution

            def logging_policy(params, observations, key_sample):
                param_subset = (params[0], params[1])
                logits = policy_network.apply(*param_subset, observations)
                if deterministic:
                    return parametric_action_distribution.mode(logits), {}
                raw_actions = parametric_action_distribution.sample_no_postprocessing(
                    logits, key_sample
                )
                postprocessed_actions = parametric_action_distribution.postprocess(
                    raw_actions
                )
                return postprocessed_actions, {}

            return logging_policy

        return make_logging_policy

    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    # Define policy_params_fn for logging during training
    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy  # Unused, use our custom logging inference

        rollout_logging_fn(
            env=eval_env,
            jit_reset=jit_reset,
            jit_step=jit_step,
            jit_inference_fn=lambda p, obs, rng: jit_logging_inference_fn(p, obs, rng),
            params=params,
            current_step=current_step,
            model_path=model_path,
            episode_length=cfg.train.episode_length,
            render_camera=cfg.logging.render_camera,
            render_fps=cfg.logging.render_fps,
        )

    # Build training config
    training_params = {
        "num_timesteps": cfg.train.num_timesteps,
        "episode_length": cfg.train.episode_length,
        "num_envs": cfg.train.num_envs,
        "batch_size": cfg.train.batch_size,
        "num_minibatches": cfg.train.num_minibatches,
        "num_updates_per_batch": cfg.train.num_updates_per_batch,
        "learning_rate": cfg.train.learning_rate,
        "entropy_cost": cfg.train.entropy_cost,
        "discounting": cfg.train.discounting,
        "reward_scaling": cfg.train.reward_scaling,
        "normalize_observations": cfg.train.normalize_observations,
        "unroll_length": cfg.train.unroll_length,
        "action_repeat": cfg.train.get("action_repeat", 1),
        "seed": cfg.train.seed,
    }

    num_evals = int(cfg.train.num_timesteps / cfg.train.eval_every)

    # Create training function
    train_fn = functools.partial(
        brax_ppo.train,
        **training_params,
        num_evals=num_evals,
        network_factory=network_factory,
        restore_checkpoint_path=None,
        progress_fn=wandb_progress,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training),
    )

    # Run training
    logging.info("Starting training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        policy_params_fn=functools.partial(
            policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
        ),
    )

    logging.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
