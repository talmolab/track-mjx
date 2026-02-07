"""Standard Brax PPO training on any vnl-playground environment.

Trains an end-to-end MLP policy using standard Brax PPO on any registered
vnl-playground task. No pretrained decoder or high-level transfer — just
direct policy optimization.

Usage:
    # Basic usage (any registered task)
    python train_task.py --task RodentBowlEscape
    python train_task.py --task RodentRearing

    # With PPO overrides
    python train_task.py --task RodentRearing --num_timesteps 1e8 --entropy_cost 0.1

    # With env config overrides (dot notation for nested)
    python train_task.py --task RodentBowlEscape --env "target_speed=1.5 ctrl_dt=0.02"

    # With custom observation keys
    python train_task.py --task RodentBowlEscape \\
        --policy_obs_key state --value_obs_key privileged_state
"""

import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import functools
import json
from datetime import datetime
from typing import Any

import imageio
import jax
import jax.numpy as jp
import wandb
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from mujoco_playground import wrapper
from orbax import checkpoint as ocp

from vnl_playground import registry
from track_mjx.scripts.utils import apply_env_overrides, parse_env_overrides_str

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# Default PPO parameters (task-independent)
DEFAULT_PPO_PARAMS = {
    "num_timesteps": int(3e8),
    "reward_scaling": 1.0,
    "episode_length": 1000,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 20,
    "num_minibatches": 16,
    "num_updates_per_batch": 4,
    "discounting": 0.99,
    "learning_rate": 1e-4,
    "entropy_cost": 0.01,
    "num_envs": 4096,
    "batch_size": 1024,
    "max_grad_norm": 1.0,
    "eval_every": 5_000_000,
}


def create_ppo_params(args: argparse.Namespace):
    """Create PPO parameters from defaults and CLI overrides."""
    params = dict(DEFAULT_PPO_PARAMS)

    # Add network factory (not in DEFAULT_PPO_PARAMS since it's a nested config)
    params["network_factory"] = config_dict.create(
        policy_hidden_layer_sizes=(1024, 512, 256),
        value_hidden_layer_sizes=(1024, 512, 256),
    )

    # Apply CLI overrides
    if args.num_timesteps is not None:
        params["num_timesteps"] = int(float(args.num_timesteps))
    if args.entropy_cost is not None:
        params["entropy_cost"] = args.entropy_cost
    if args.episode_length is not None:
        params["episode_length"] = args.episode_length
    if args.eval_every is not None:
        params["eval_every"] = args.eval_every
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.num_envs is not None:
        params["num_envs"] = args.num_envs

    return config_dict.create(**params)


def create_environments(task_name: str, env_cfg: Any):
    """Create training and eval environments."""
    env = registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False)
    eval_env = registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False)
    return env, eval_env


def make_logging_inference_fn(ppo_networks):
    """Creates inference function for eval rollouts."""

    def make_logging_policy(deterministic: bool = False):
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(params, observations, key_sample):
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, observations)
            if deterministic:
                return (
                    jp.array(ppo_networks.parametric_action_distribution.mode(logits)),
                    {},
                )
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return jp.array(postprocessed_actions), {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return logging_policy

    return make_logging_policy


def create_policy_params_fn(
    ppo_params,
    ckpt_path: epath.Path,
    env,
    jit_reset,
    jit_step,
    jit_logging_inference_fn,
):
    """Create the policy_params_fn for evaluation and checkpointing."""

    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy

        steps_k = current_step * ppo_params.eval_every / 1000
        wandb.log({"train/steps_k": steps_k}, commit=False)

        # Generate rollout with randomized initial state based on current_step
        rng = jax.random.PRNGKey(current_step)
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
        rollout = [state]
        for _ in range(int(ppo_params.episode_length)):
            _, rng = jax.random.split(rng)
            action, _ = jit_logging_inference_fn(params, state.obs, rng)
            state = jit_step(state, action)
            rollout.append(state)

        # Render and save video
        video_path = f"{ckpt_path}/{current_step}.mp4"
        frames = env.render(rollout, camera="close_profile-rodent")
        imageio.mimsave(video_path, frames, fps=int(1.0 / float(env.dt)))
        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")}, commit=False)

        # Save checkpoint
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    return functools.partial(
        policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Standard Brax PPO training on any vnl-playground environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_task.py --task RodentBowlEscape

    python train_task.py --task RodentRearing --num_timesteps 1e8 --entropy_cost 0.1

    python train_task.py --task RodentBowlEscape \\
        --env "target_speed=1.5 reward_terms.head_height_dense.weight=0.0"
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task environment name (any task registered in vnl-playground)",
    )

    # PPO overrides
    parser.add_argument(
        "--num_timesteps",
        type=str,
        default=None,
        help="Override num_timesteps (e.g., 3e8)",
    )
    parser.add_argument(
        "--entropy_cost", type=float, default=None, help="Override entropy_cost"
    )
    parser.add_argument(
        "--episode_length", type=int, default=None, help="Override episode_length"
    )
    parser.add_argument(
        "--eval_every", type=int, default=None, help="Override eval frequency"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs")

    # Env config overrides
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help='Env config overrides (space-separated key=value, e.g., "target_speed=1.0")',
    )
    parser.add_argument(
        "--policy_obs_key",
        type=str,
        default="state",
        help="Observation key for policy network (default: state)",
    )
    parser.add_argument(
        "--value_obs_key",
        type=str,
        default="privileged_state",
        help="Observation key for value network (default: privileged_state)",
    )

    # Output options
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="task_checkpoints",
        help="Base checkpoint directory",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vnl-playground",
        help="Wandb project name",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    print(f"Task: {args.task}")

    # Create configs
    cli_env_overrides = parse_env_overrides_str(args.env)
    env_cfg = registry.get_default_config(args.task)
    apply_env_overrides(env_cfg, cli_env_overrides)
    ppo_params = create_ppo_params(args)
    print(f"env_cfg:\n{env_cfg}")
    print(f"ppo_params:\n{ppo_params}")

    # Setup experiment (include microseconds to avoid run ID collisions)
    exp_name = f"{args.task}-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    ckpt_path = epath.Path(args.checkpoint_dir).resolve() / exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment name: {exp_name}")
    print(f"Checkpoint path: {ckpt_path}")

    # Save config
    config_to_save = {
        "task": args.task,
        "env_config": env_cfg.to_dict(),
        "ppo_params": dict(ppo_params),
        "cli_env_overrides": cli_env_overrides,
        "policy_obs_key": args.policy_obs_key,
        "value_obs_key": args.value_obs_key,
    }
    with open(ckpt_path / "config.json", "w") as fp:
        json.dump(config_to_save, fp, indent=4, default=lambda o: str(o))

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=config_to_save,
        id=f"task-ppo-{exp_name}",
        notes=f"task: {args.task}",
    )

    def wandb_progress(num_steps, metrics):
        print(f"Step {num_steps}")
        print(f"Metrics: {metrics}")
        wandb.log(metrics)

    # Create environments
    env, eval_env = create_environments(args.task, env_cfg)

    # Setup training
    training_params = dict(ppo_params)
    del training_params["network_factory"]
    del training_params["eval_every"]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_obs_key=args.policy_obs_key,
        value_obs_key=args.value_obs_key,
        **ppo_params.network_factory,
    )
    normalize = lambda x, _y: x
    if training_params["normalize_observations"]:
        normalize = running_statistics.normalize

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        num_evals=int(ppo_params.num_timesteps / ppo_params.eval_every),
        network_factory=network_factory,
        restore_checkpoint_path=None,
        progress_fn=wandb_progress,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training),
    )

    # Setup logging
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)

    # Get observation size from dict structure (policy obs key)
    obs_size = start_state.obs[args.policy_obs_key].shape[-1]

    ppo_network = network_factory(
        obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    policy_params_fn = create_policy_params_fn(
        ppo_params, ckpt_path, env, jit_reset, jit_step, jit_logging_inference_fn
    )

    # Run training
    train_fn(environment=env, eval_env=eval_env, policy_params_fn=policy_params_fn)
    print("Training complete!")


if __name__ == "__main__":
    main()
