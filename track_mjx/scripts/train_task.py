"""Standard Brax PPO training on any vnl-playground environment.

Trains an end-to-end MLP policy using standard Brax PPO on any registered
vnl-playground task. No pretrained decoder or high-level transfer — just
direct policy optimization.

Supports both JAX/MJX (default) and Warp backends. For Warp, pass
--env "mujoco_impl=warp" to enable the full-collision rodent model.

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

    # Warp backend (full-collision rodent model)
    python train_task.py --task RodentBowlEscape --env "mujoco_impl=warp"
    python train_task.py --task RodentBowlEscape --env "mujoco_impl=warp" \\
        --naconmax_per_env 90 --njmax 1200
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

import jax
import wandb
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from mujoco_playground import wrapper

from vnl_playground import registry
from track_mjx.scripts.utils import (
    FlattenInnerObsWrapper,
    apply_env_overrides,
    configure_warp_backend,
    create_base_parser,
    create_policy_params_fn,
    create_ppo_params,
    make_logging_inference_fn,
    parse_env_overrides_str,
    setup_jax_cache,
)

setup_jax_cache()


def create_environments(task_name, env_cfg):
    """Create training and eval environments."""
    env = FlattenInnerObsWrapper(
        registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False)
    )
    eval_env = FlattenInnerObsWrapper(
        registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False)
    )
    return env, eval_env


def parse_args():
    """Parse command-line arguments."""
    parser = create_base_parser(
        description="Standard Brax PPO training on any vnl-playground environment.",
        epilog="""
Examples:
    python train_task.py --task RodentBowlEscape

    python train_task.py --task RodentRearing --num_timesteps 1e8 --entropy_cost 0.1

    python train_task.py --task RodentBowlEscape \\
        --env "target_speed=1.5 reward_terms.head_height_dense.weight=0.0"

    # Warp backend (full-collision rodent model)
    python train_task.py --task RodentBowlEscape --env "mujoco_impl=warp"
        """,
    )

    # Warp-specific args
    parser.add_argument(
        "--naconmax_per_env",
        type=int,
        default=90,
        help="Max active contacts per env for Warp backend (default: 90). "
        "naconmax is computed as num_envs * naconmax_per_env.",
    )
    parser.add_argument(
        "--njmax",
        type=int,
        default=1200,
        help="Max constraint rows per env for Warp backend (default: 1200).",
    )

    args = parser.parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = "task_checkpoints"
    return args


def main():
    """Main training entry point."""
    args = parse_args()
    print(f"Task: {args.task}")

    # Create configs
    cli_env_overrides = parse_env_overrides_str(args.env)
    env_cfg = registry.get_default_config(args.task)
    apply_env_overrides(env_cfg, cli_env_overrides)

    # Detect Warp backend and auto-configure
    is_warp = getattr(env_cfg, "mujoco_impl", None) == "warp"
    default_num_envs = 1024 if is_warp else 4096
    ppo_params = create_ppo_params(args, default_num_envs=default_num_envs)

    if is_warp:
        configure_warp_backend(
            env_cfg, ppo_params.num_envs, args.naconmax_per_env, args.njmax
        )

    print(f"env_cfg:\n{env_cfg}")
    print(f"ppo_params:\n{ppo_params}")

    # Setup experiment (include microseconds to avoid run ID collisions)
    backend_tag = "warp-ppo" if is_warp else "ppo"
    exp_name = (
        f"{args.task}-{backend_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    )
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
        "seed": args.seed,
    }
    if is_warp:
        config_to_save["backend"] = "warp"
        config_to_save["naconmax_per_env"] = args.naconmax_per_env
        config_to_save["njmax"] = args.njmax
    with open(ckpt_path / "config.json", "w") as fp:
        json.dump(config_to_save, fp, indent=4, default=lambda o: str(o))

    # Initialize wandb
    wandb_id = f"task-{backend_tag}-{exp_name}"
    wandb.init(
        project=args.wandb_project,
        config=config_to_save,
        id=wandb_id,
        notes=f"task: {args.task}" + (" (warp, full-collision)" if is_warp else ""),
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
        seed=args.seed,
        num_evals=int(ppo_params.num_timesteps / ppo_params.eval_every),
        network_factory=network_factory,
        restore_checkpoint_path=None,
        progress_fn=wandb_progress,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training),
    )

    # Setup logging
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
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
        ppo_params, ckpt_path, env, jit_reset, jit_step, jit_logging_inference_fn
    )

    # Run training
    train_fn(environment=env, eval_env=eval_env, policy_params_fn=policy_params_fn)
    print("Training complete!")


if __name__ == "__main__":
    main()
