import os

# os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

# set default env variable if not set
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"
)
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True "
)

# from jax.experimental.compilation_cache import compilation_cache as cc

# cc.set_cache_dir("/tmp/jax_cache")

import functools
import json
from datetime import datetime

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from IPython.display import clear_output, display
from orbax import checkpoint as ocp

from mujoco_playground import locomotion, wrapper
from mujoco_playground.config import locomotion_params

# # Enable persistent compilation cache.
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
# )

env_name = "Go1JoystickRoughTerrain"
env_cfg = locomotion.get_default_config(env_name)
randomizer = locomotion.get_domain_randomizer(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

from track_mjx.environment.task import joysticks

env = joysticks.RodentJoystick()

from pprint import pprint

ppo_params.num_evals = 500
ppo_params.num_timesteps = 1_000_000_000
ppo_params.num_envs = 2048
pprint(ppo_params)

# Setup wandb logging.
USE_WANDB = True

if USE_WANDB:
    wandb.init(project="rodent_joysticks", config=ppo_params.to_dict())
    wandb.config.update(
        {
            "env_name": "rodent_joysticks",
        }
    )
    wandb.run.name = (
        f"rodent_joysticks_{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    

SUFFIX = None
FINETUNE_PATH = None
env_name = "rodent_joysticks"
# Generate unique experiment name.
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
if SUFFIX is not None:
    exp_name += f"-{SUFFIX}"
print(f"Experiment name: {exp_name}")

# Possibly restore from the latest checkpoint.
if FINETUNE_PATH is not None:
    FINETUNE_PATH = epath.Path(FINETUNE_PATH)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    latest_ckpt = latest_ckpts[-1]
    restore_checkpoint_path = latest_ckpt
    print(f"Restoring from: {restore_checkpoint_path}")
else:
    restore_checkpoint_path = None

ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint path: {ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    # Log to wandb.
    if USE_WANDB:
        wandb.log(metrics, step=num_steps)


def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


training_params = dict(ppo_params)
del training_params["network_factory"]

train_fn = functools.partial(
    ppo.train,
    **training_params,
    network_factory=functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    ),
    restore_checkpoint_path=restore_checkpoint_path,
    progress_fn=progress,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    policy_params_fn=policy_params_fn,
)

env = joysticks.RodentJoystick()
eval_env = joysticks.RodentJoystick(evaluator=True)
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
if len(times) > 1:
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
