"""Shared utilities for training scripts."""

import argparse
import functools
from typing import Any

import imageio
import jax
import jax.numpy as jp
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp
from vnl_playground.tasks.rodent import consts

import wandb


def parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "none":
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def apply_env_overrides(env_cfg: Any, overrides: dict) -> None:
    """Apply overrides to environment config using dot notation."""
    for key, value in overrides.items():
        parts = key.split(".")
        obj = env_cfg
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        except AttributeError as e:
            raise ValueError(
                f"Invalid env config override '{key}={value}': {e}. "
                f"Check that the config path exists."
            ) from e


def parse_env_overrides_str(overrides_str: str | None) -> dict:
    """Parse space-separated key=value overrides string."""
    if not overrides_str:
        return {}
    result = {}
    for kv in overrides_str.split():
        if "=" not in kv:
            raise ValueError(
                f"Invalid env override token '{kv}': expected 'key=value' format. "
                f"Full override string: '{overrides_str}'"
            )
        key, value = kv.split("=", 1)
        result[key] = parse_value(value)
    return result


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


def get_default_ppo_params(num_envs: int = 4096) -> dict:
    """Return a fresh copy of default PPO params with the given num_envs."""
    params = dict(DEFAULT_PPO_PARAMS)
    params["num_envs"] = num_envs
    return params


def create_ppo_params(args: argparse.Namespace, default_num_envs: int = 4096):
    """Create PPO ConfigDict from defaults and CLI overrides.

    Parses network sizes from args.policy_hidden_sizes / args.value_hidden_sizes,
    then applies all scalar CLI overrides.
    """
    params = get_default_ppo_params(default_num_envs)

    policy_sizes = tuple(int(x) for x in args.policy_hidden_sizes.split(","))
    value_sizes = tuple(int(x) for x in args.value_hidden_sizes.split(","))
    params["network_factory"] = config_dict.create(
        policy_hidden_layer_sizes=policy_sizes,
        value_hidden_layer_sizes=value_sizes,
    )

    override_keys = [
        "entropy_cost",
        "episode_length",
        "eval_every",
        "learning_rate",
        "num_envs",
        "unroll_length",
        "discounting",
        "batch_size",
    ]
    for key in override_keys:
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if args.num_timesteps is not None:
        params["num_timesteps"] = int(float(args.num_timesteps))

    return config_dict.create(**params)


def get_training_params(ppo_params):
    """Extract Brax PPO training params by removing non-training keys."""
    training_params = dict(ppo_params)
    del training_params["network_factory"]
    del training_params["eval_every"]
    return training_params


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
        wandb.log({"training/steps_k": steps_k}, commit=False)

        # Generate rollout with randomized initial state based on current_step
        rng = jax.random.key(current_step)
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
        camera = getattr(env, "_default_render_camera", -1)
        frames = env.render(rollout, camera=camera)
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


def create_base_parser(
    description: str, epilog: str | None = None
) -> argparse.ArgumentParser:
    """Create an ArgumentParser with the common args shared across scripts.

    Includes: --task, all PPO overrides, --seed, network architecture,
    --env, --policy_obs_key, --value_obs_key, --checkpoint_dir, --wandb_project.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
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
    parser.add_argument(
        "--unroll_length", type=int, default=None, help="Override unroll_length"
    )
    parser.add_argument(
        "--discounting", type=float, default=None, help="Override discounting"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch_size"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # Network architecture
    parser.add_argument(
        "--policy_hidden_sizes",
        type=str,
        default="1024,512,256",
        help="Comma-separated policy hidden layer sizes (default: 1024,512,256)",
    )
    parser.add_argument(
        "--value_hidden_sizes",
        type=str,
        default="1024,512,256",
        help="Comma-separated value hidden layer sizes (default: 1024,512,256)",
    )

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
        default="state",
        help="Observation key for value network (default: state)",
    )

    # Output options
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Base checkpoint directory",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vnl-playground",
        help="Wandb project name",
    )

    return parser


def configure_warp_backend(
    env_cfg: Any,
    num_envs: int,
    task_name: str = "",
) -> Any:
    """Configure env_cfg for the Warp backend.

    For rodent tasks, sets walker_xml_path to the full-collision model.
    """
    is_rodent = task_name.lower().startswith("rodent")
    if is_rodent:
        env_cfg.walker_xml_path = consts.RODENT_NO_TAIL_COLLISION_XML
    walker_info = "rodent_no_tail_collisions.xml" if is_rodent else "default (env)"
    print(f"Warp backend: walker={walker_info}")
    return env_cfg
