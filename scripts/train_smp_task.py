"""Train a VNL rodent task with a frozen SMP prior reward."""

import os
import sys

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
from dataclasses import replace
from datetime import datetime

import jax
import wandb
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from mujoco_playground import wrapper
from vnl_playground import registry
from vnl_playground.tasks import wrappers as rodent_wrappers

from track_mjx.agent.smp.checkpointing import load_prior
from track_mjx.agent.smp.wrappers import SMPRewardWrapper
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


def create_environments(task_name, env_cfg, prior_bundle, reward_config):
    """Create SMP-wrapped training and eval environments."""

    wrapper_kwargs = dict(
        prior_params=prior_bundle["params"],
        prior_normalizer=prior_bundle["normalizer"],
        diff_normalizer=prior_bundle["diff_normalizer"],
        model_config=prior_bundle["model_config"],
        feature_spec=prior_bundle["feature_spec"],
        reward_config=reward_config,
        metadata=prior_bundle["metadata"],
    )
    env = rodent_wrappers.BraxObsWrapper(
        SMPRewardWrapper(
            registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False),
            **wrapper_kwargs,
        )
    )
    eval_env = rodent_wrappers.BraxObsWrapper(
        SMPRewardWrapper(
            registry.load(task_name, config=env_cfg, clips=None, flatten_obs=False),
            **wrapper_kwargs,
        )
    )
    return env, eval_env


def parse_args():
    parser = create_base_parser(
        description="Train a VNL rodent task with SMP prior reward.",
        epilog="""
Examples:
    python scripts/train_smp_task.py --task RodentMaintainVelocity --smp_prior outputs/smp_prior/latest
    python scripts/train_smp_task.py --task RodentBowlEscape --smp_prior outputs/smp_prior/latest --env "target_speed=1.5"
        """,
    )
    parser.add_argument("--smp_prior", required=True, help="Path to trained SMP prior.")
    parser.add_argument(
        "--no_ema", action="store_true", help="Use non-EMA prior params."
    )
    parser.add_argument("--task_reward_weight", type=float, default=None)
    parser.add_argument("--smp_reward_weight", type=float, default=None)
    parser.add_argument("--sds_loss_scale", type=float, default=None)
    parser.add_argument("--smp_reward_scale", type=float, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = "smp_task_checkpoints"
    return args


def main():
    args = parse_args()
    print(f"Task: {args.task}")
    print(f"SMP prior: {args.smp_prior}")

    prior_bundle = load_prior(args.smp_prior, use_ema=not args.no_ema)
    reward_config = prior_bundle["reward_config"]
    if args.task_reward_weight is not None:
        reward_config = replace(
            reward_config, task_reward_weight=args.task_reward_weight
        )
    if args.smp_reward_weight is not None:
        reward_config = replace(reward_config, smp_reward_weight=args.smp_reward_weight)
    if args.sds_loss_scale is not None:
        reward_config = replace(reward_config, sds_loss_scale=args.sds_loss_scale)
    if args.smp_reward_scale is not None:
        reward_config = replace(reward_config, smp_reward_scale=args.smp_reward_scale)

    cli_env_overrides = parse_env_overrides_str(args.env)
    env_cfg = registry.get_default_config(args.task)
    is_warp = cli_env_overrides.get("mujoco_impl") == "warp"
    default_num_envs = 1024 if is_warp else 4096
    ppo_params = create_ppo_params(args, default_num_envs=default_num_envs)

    if is_warp:
        env_cfg.mujoco_impl = "warp"
        configure_warp_backend(env_cfg, ppo_params.num_envs, task_name=args.task)
    apply_env_overrides(env_cfg, cli_env_overrides)

    backend_tag = "warp-smp" if is_warp else "smp"
    exp_name = (
        f"{args.task}-{backend_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    ckpt_path = epath.Path(args.checkpoint_dir).resolve() / exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment name: {exp_name}")
    print(f"Checkpoint path: {ckpt_path}")

    config_to_save = {
        "task": args.task,
        "env_config": env_cfg.to_dict(),
        "ppo_params": dict(ppo_params),
        "smp_prior": str(epath.Path(args.smp_prior).resolve()),
        "smp_reward": reward_config.to_dict(),
        "smp_metadata": prior_bundle["metadata"],
        "cli_env_overrides": cli_env_overrides,
        "policy_obs_key": args.policy_obs_key,
        "value_obs_key": args.value_obs_key,
        "seed": args.seed,
    }
    with open(ckpt_path / "config.json", "w") as fp:
        json.dump(config_to_save, fp, indent=4, default=lambda o: str(o))

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            config=config_to_save,
            id=f"smp-{exp_name}",
            notes=f"task: {args.task}, smp_prior: {args.smp_prior}",
        )

    def wandb_progress(num_steps, metrics):
        print(f"Step {num_steps}")
        print(f"Metrics: {metrics}")
        wandb.log(metrics)

    env, eval_env = create_environments(args.task, env_cfg, prior_bundle, reward_config)
    training_params = get_training_params(ppo_params)
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

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    start_state = jit_reset(jax.random.PRNGKey(args.seed))
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

    train_fn(environment=env, eval_env=eval_env, policy_params_fn=policy_params_fn)
    print("Training complete!")


if __name__ == "__main__":
    main()
