"""Shared utilities for training scripts."""

import argparse
import functools
from typing import Any, Optional

import imageio
import jax
import jax.numpy as jp
import wandb
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp
from vnl_playground.tasks.rodent import consts
from track_mjx.agent.wandb_logging import log_lineplot_to_wandb


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


def parse_env_overrides_str(overrides_str: Optional[str]) -> dict:
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


# ---------------------------------------------------------------------------
# JAX cache setup
# ---------------------------------------------------------------------------


def setup_jax_cache():
    """Enable persistent JAX compilation cache."""
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ---------------------------------------------------------------------------
# PPO parameter helpers
# ---------------------------------------------------------------------------

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

    # Network factory
    policy_sizes = tuple(int(x) for x in args.policy_hidden_sizes.split(","))
    value_sizes = tuple(int(x) for x in args.value_hidden_sizes.split(","))
    distribution_type = args.distribution_type
    state_dependent_std = False
    init_noise_std = args.init_noise_std
    if distribution_type == "normal":
        state_dependent_std = True
    params["network_factory"] = config_dict.create(
        policy_hidden_layer_sizes=policy_sizes,
        value_hidden_layer_sizes=value_sizes,
        distribution_type=distribution_type,
        state_dependent_std=state_dependent_std,
        init_noise_std=init_noise_std,
    )

    # Apply CLI overrides (None means "use default")
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


# ---------------------------------------------------------------------------
# Logging / evaluation helpers
# ---------------------------------------------------------------------------


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

    def policy_params_fn(
        current_step,
        make_policy,
        params,
        jit_logging_inference_fn,
        rollout_metrics=None,
    ):
        del make_policy

        steps_k = current_step * ppo_params.eval_every / 1000
        wandb.log({"training/steps_k": steps_k}, commit=False)

        # Generate rollout with randomized initial state based on current_step
        rng = jax.random.PRNGKey(current_step)
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
        rollout = [state]
        latent_means = []
        ctrls = []
        for _ in range(int(ppo_params.episode_length)):
            _, rng = jax.random.split(rng)
            action, extras = jit_logging_inference_fn(params, state.obs, rng)
            state = jit_step(state, action)
            rollout.append(state)
            latent_means.append(action)
            ctrls.append(state.info["action"])

        latent_means = jp.stack(latent_means)

        means_mean = jp.mean(latent_means, axis=0)
        means_std = jp.std(latent_means, axis=0)

        ctrls = jp.stack(ctrls)
        ctrls_mean = jp.mean(ctrls, axis=0)
        ctrls_std = jp.std(ctrls, axis=0)

        if rollout_metrics is not None:
            for metric in rollout_metrics:
                if metric.lower() == "latents":
                    for i in range(means_mean.shape[0]):
                        wandb.log(
                            {
                                f"latents/latent_means_mean{i}": means_mean[i],
                                f"latents/latent_means_std{i}": means_std[i],
                            },
                            commit=False,
                        )
                        log_lineplot_to_wandb(
                            name=f"latents/latent_means{i}",
                            metric_name=f"latent_means{i}",
                            data=list(enumerate(latent_means[:, i])),
                            title="Latent mean per rollout frame",
                        )
                elif metric.lower() == "ctrl":
                    for i in range(ctrls_mean.shape[0]):
                        wandb.log(
                            {
                                f"ctrls/ctrls_mean{i}": ctrls_mean[i],
                                f"ctrls/ctrls_std{i}": ctrls_std[i],
                            },
                            commit=False,
                        )
                        log_lineplot_to_wandb(
                            name="ctrls/ctrls{i}",
                            metric_name=f"ctrls{i}",
                            data=list(enumerate(ctrls[:, i])),
                            title="Ctrl per rollout frame",
                        )
                else:
                    if metric in rollout[0].metrics:
                        log_lineplot_to_wandb(
                            name=f"eval/rollout_{metric}",
                            metric_name=metric,
                            data=list(enumerate([s.metrics[metric] for s in rollout])),
                            title=f"{metric} per rollout frame",
                        )
                    else:
                        print(
                            f"Rollout metric '{metric}' not found in environment metrics."
                        )

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


# ---------------------------------------------------------------------------
# Shared argparse
# ---------------------------------------------------------------------------


def create_base_parser(
    description: str, epilog: Optional[str] = None
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
    parser.add_argument(
        "--distribution_type",
        type=str,
        default="tanh_normal",
        help="Distribution type (default: tanh_normal)",
        choices=["tanh_normal", "normal"],
    )
    parser.add_argument(
        "--init_noise_std",
        type=float,
        default=1.0,
        help="Initial noise std (default: 0.0)",
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


# ---------------------------------------------------------------------------
# Warp backend configuration
# ---------------------------------------------------------------------------


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
