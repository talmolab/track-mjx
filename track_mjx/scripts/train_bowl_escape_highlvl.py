"""High-level training for RodentBowlEscape task.

DEPRECATED: Use train_highlvl.py instead:
    python train_highlvl.py --task=RodentBowlEscape --mimic_checkpoint=260115_005843_966729
"""

import os
import warnings
from typing import Callable, Mapping

warnings.warn(
    "This script is deprecated. Use train_highlvl.py instead:\n"
    "  python train_highlvl.py --task=RodentBowlEscape --mimic_checkpoint=...",
    DeprecationWarning,
    stacklevel=2,
)

from omegaconf import OmegaConf

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
from datetime import datetime
import numpy as np
import imageio

import jax
import jax.numpy as jp

import mujoco
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics

from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from ml_collections import config_dict

from mujoco_playground import wrapper

from vnl_playground import registry
from vnl_playground.tasks.rodent import bowl_escape
from vnl_playground.tasks import wrappers as rodent_wrappers
import hydra
from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

env_cfg = bowl_escape.default_config()
mimic_run_id = "260115_005843_966729"  # Replace with your checkpoint run ID
mimic_checkpoint_path = hydra.utils.to_absolute_path(
    f"./model_checkpoints/{mimic_run_id}"
)
mimic_cfg = OmegaConf.create(
    checkpointing.load_config_from_checkpoint(mimic_checkpoint_path)
)

env_cfg.ctrl_dt = mimic_cfg.env_config.ctrl_dt
env_cfg.target_speed = 1.0

# TODO: support recurrent decoders
decoder_policy_fn = ff_ppo_networks.make_decoder_policy_fn(mimic_checkpoint_path)
print(f"env_cfg:\n{env_cfg}")


ppo_params = config_dict.create(
    num_timesteps=int(3e8),  # 300 million
    reward_scaling=1.0,
    episode_length=1500,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=16,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=1e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=1024,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(1024, 512, 256),
        value_hidden_layer_sizes=(1024, 512, 256),
    ),
    eval_every=5_000_000,  # num_evals = num_timesteps // eval_every
)
print(f"ppo_params:\n{ppo_params}")

env_name = "RodentBowlEscape"

# Generate unique experiment name.
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"

print(f"Experiment name: {exp_name}")

ckpt_path = epath.Path("highlvl_checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint path: {ckpt_path}")

# Save config
config_to_save = {
    "env_config": env_cfg.to_dict(),
    "ppo_params": dict(ppo_params),
    "mimic_run_id": mimic_run_id,
}
with open(ckpt_path / "config.json", "w") as fp:
    json.dump(config_to_save, fp, indent=4, default=lambda o: str(o))

# Initialize wandb with combined config
wandb.init(
    project="vnl-playground",
    config=config_to_save,
    id=f"highlvl-{exp_name}",
    notes=f"mimic run: {mimic_run_id}",
)


def wandb_progress(num_steps, metrics):
    print(f"Step {num_steps}")
    print(f"Metrics: {metrics}")
    wandb.log(metrics)


training_params = dict(ppo_params)
del training_params["network_factory"]
del training_params["eval_every"]

# stuff to make logging inference fn in this file
network_factory = functools.partial(
    ppo_networks.make_ppo_networks, **ppo_params.network_factory
)
normalize = lambda x, y: x
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


def make_logging_inference_fn(ppo_networks):
    """Creates params and inference function for the PPO agent.
    The policy takes the params as an input, so different sets of params can be used.
    """

    def make_logging_policy(deterministic=False):
        policy_network = ppo_networks.policy_network
        # can modify this to provide stochastic action + noise
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params,
            observations,
            key_sample,
        ):
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, observations)
            # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)
            if deterministic:
                return (
                    jp.array(ppo_networks.parametric_action_distribution.mode(logits)),
                    {},
                )
            # action sampling is happening here, according to distribution parameter logits
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            # probability of selection specific action, actions with higher reward should have higher probability
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


if __name__ == "__main__":
    # Load with dict observations - HighLevelWrapper will extract task_obs for
    # the high-level policy and proprioception for the decoder
    base_env = registry.load(env_name, config=env_cfg, clips=None, flatten_obs=False)
    env = rodent_wrappers.HighLevelWrapper(
        base_env,
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        highlvl_obs_key="task_obs",
        decoder_obs_key="proprioception",
    )
    eval_base_env = registry.load(
        env_name, config=env_cfg, clips=None, flatten_obs=False
    )
    eval_env = rodent_wrappers.HighLevelWrapper(
        eval_base_env,
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        highlvl_obs_key="task_obs",
        decoder_obs_key="proprioception",
    )

    # Render a rollout in the policy_params_fn to log to wandb at each step
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)
    ppo_network = network_factory(
        start_state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy  # Unused.

        # Log approximate total steps in thousands
        steps_k = current_step * ppo_params.eval_every / 1000
        wandb.log({"train/steps_k": steps_k}, commit=False)

        # Generate a rollout
        rollout = [start_state]
        state = start_state
        rng = jax.random.PRNGKey(0)
        for _ in range(ppo_params.episode_length):
            _, rng = jax.random.split(rng)
            action, _ = jit_logging_inference_fn(params, state.obs, rng)
            state = jit_step(state, action)
            rollout.append(state)

        # render and log
        video_path = f"{ckpt_path}/{current_step}.mp4"
        frames = env.render(rollout, camera="close_profile-rodent")
        imageio.mimsave(video_path, frames, fps=int((1.0 / env.dt)))
        # don't commit because progress_fn is called after
        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")}, commit=False)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    # only run the training if this file is run as a script
    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        policy_params_fn=functools.partial(
            policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
        ),
    )
