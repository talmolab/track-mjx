import argparse
import subprocess
import sys


"""
example commands:
python slurm.py
python slurm.py --config-name=rodent-recurrent.yaml --time=2-00:00
python slurm.py --task=bowl_escape_transfer --config_name=bowl_escape_transfer network_config.latent_ar1_weight=0.0

Standard Brax PPO on any vnl-playground task (use -- separator):
python slurm.py --task=task_ppo -- --task=RodentBowlEscape
python slurm.py --task=task_ppo -- --task=RodentRearing --num_timesteps=1e8

Warp backend PPO (via --env flag on train_task.py):
python slurm.py --task=task_ppo -- --task=RodentBowlEscape --env "mujoco_impl=warp"
python slurm.py --task=task_ppo_warp -- --task=RodentBowlEscape  # backward compat alias

High-level training (unified script - any vnl-playground task):
python slurm.py --task=highlvl -- --task=RodentBowlEscape --mimic_checkpoint=260210_013247_285744
python slurm.py --task=highlvl -- --task=RodentJoystick --mimic_checkpoint=260210_013247_285744
python slurm.py --task=highlvl -- --task=RodentRearing --mimic_checkpoint=260210_013247_285744 --entropy_cost=0.1
python slurm.py --task=highlvl -- --task=MyCustomTask --mimic_checkpoint=260210_013247_285744

High-level training with Warp backend (full-collision model):
python slurm.py --task=highlvl -- --task=RodentBowlEscape --mimic_checkpoint=260131_223134_344901 --env "mujoco_impl=warp" --num_envs=1024

Legacy scripts (for clip-based tasks or backward compatibility):
python slurm.py --task=bowl_escape_highlvl
python slurm.py --task=rodent_rear_highlvl
python slurm.py --task=sparse_imitation_highlvl  # clip-based
python slurm.py --task=imitation_highlvl  # clip-based

With V0:
python slurm.py --gpu_type=anybig --num_gpus=1 --time=1-00:00 --task=tracking --config_name=charles_1gpu

with hydra overrides (use -- to separate slurm args from hydra overrides):
python slurm.py --gpu_type=a100 --num_gpus=1 -- train_setup.num_timesteps=1e9 loss.kl_coef=0.01
python slurm.py --config_name=charles_1gpu -- model.hidden_size=512 train_setup.learning_rate=3e-4

training on ssh'd v100:
python scripts/train.py --config-name=charles_1v100
"""


def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(
            ["sbatch"], input=script, universal_newlines=True
        )
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def submit(
    task,
    gpu_type,
    num_gpus,
    job_name,
    mem,
    cpus,
    time,
    out_dir,
    config_name,
    hydra_overrides,
):
    """
    Construct and submit the SLURM script with the specified parameters.
    """
    # Define GPU configurations
    gpu_configs = {
        "a100": "nvidia_a100-sxm4-40gb",
        "h100": "nvidia_h100_80gb_hbm3",
        "a40": "nvidia_a40",
        "h200": "nvidia_h200",
    }
    if gpu_type in gpu_configs.keys():
        gpu_resource = f"gpu:{gpu_configs[gpu_type]}:{num_gpus}"
    elif gpu_type == "anybig":
        gpu_resource = f"""gpu:{num_gpus}
#SBATCH --constraint=\"a100|h200|h100\"
"""
    else:
        raise ValueError(f"Invalid GPU type: {gpu_type}")

    tasks = {
        "tracking": "scripts/train.py",
        "bowl_escape_transfer": "scripts/train_bowl_escape_transfer.py",
        # Unified high-level script
        "highlvl": "scripts/train_highlvl.py",
        # Standard Brax PPO on any vnl-playground task (also supports Warp via --env "mujoco_impl=warp")
        "task_ppo": "scripts/train_task.py",
        # Backward compat alias: auto-prepends --env "mujoco_impl=warp" to train_task.py
        "task_ppo_warp": "scripts/train_task.py",
        # Legacy high-level scripts (deprecated, kept for backward compatibility)
        "bowl_escape_highlvl": "scripts/train_bowl_escape_highlvl.py",
        "rodent_rear_highlvl": "scripts/train_rodent_rear_highlvl.py",
        "sparse_imitation_highlvl": "scripts/train_sparse_imitation_highlvl.py",
        "imitation_highlvl": "scripts/train_imitation_highlvl.py",
    }

    # Legacy highlvl scripts (no hydra, no args)
    legacy_highlvl_tasks = (
        "bowl_escape_highlvl",
        "rodent_rear_highlvl",
        "sparse_imitation_highlvl",
        "imitation_highlvl",
    )

    # CLI-based tasks: pass overrides directly as script args (no Hydra)
    cli_tasks = {"highlvl", "task_ppo", "task_ppo_warp"}

    if task in legacy_highlvl_tasks:
        python_cmd = f"python3 {tasks[task]}"
    elif task in cli_tasks:
        # Strip leading "--" separator if present
        args_list = [arg for arg in hydra_overrides if arg != "--"]
        # Auto-prepend Warp env override for backward compat alias
        if task == "task_ppo_warp":
            # Only add if user hasn't already specified --env with mujoco_impl
            has_env_warp = any("mujoco_impl=warp" in arg for arg in args_list)
            if not has_env_warp:
                args_list = ['--env', '"mujoco_impl=warp"'] + args_list
        overrides_str = " ".join(args_list)
        python_cmd = f"python3 {tasks[task]} {overrides_str}".rstrip()
    else:
        overrides_str = " ".join(hydra_overrides)
        python_cmd = f"python3 {tasks[task]} --config-name={config_name} {overrides_str}".rstrip()

    # Construct the SLURM script
    script = f"""#!/bin/bash
#SBATCH -p kempner,kempner_h100
#SBATCH -A kempner_pehlevan_lab
# # SBATCH -p gpu,gpu_h200
# # SBATCH -A olveczky_lab
#SBATCH --mem={mem}
#SBATCH -c {cpus}
#SBATCH -N 1
#SBATCH -t {time}
#SBATCH -J {job_name}
#SBATCH --gres={gpu_resource}
#SBATCH -o {out_dir}/%x_%j.out
#SBATCH --exclude=holygpu8a19103,holygpu8a19102

# Load necessary modules and activate environment
source ~/.bashrc
module load python
module load cuda/12.4.1-fasrc01
source .venv/bin/activate

# Display GPU information
nvidia-smi

# Env vars
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export WANDB_CACHE_DIR=$PWD
export PYTHONUNBUFFERED=1

# Add pip nvidia lib paths so jax-cuda12-plugin can find cuSPARSE, cuSOLVER, etc.
NVIDIA_LIBS=$(python3 -c "import nvidia; from pathlib import Path; print(':'.join(str(p) for p in Path(nvidia.__path__[0]).glob('*/lib') if p.is_dir()))" 2>/dev/null)
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi

# Run the training script (CLI args for highlvl/task_ppo, Hydra overrides for tracking)
{python_cmd}

"""

    print(
        f"Submitting job with GPU type: {gpu_type}, Number of GPUs: {num_gpus}, Config name: {config_name}"
    )
    if hydra_overrides:
        print(f"Hydra overrides: {' '.join(hydra_overrides)}")
    job_id = slurm_submit(script)
    print(f"Job submitted with ID: {job_id}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Submit a SLURM job with specified GPU type."
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        choices=["a100", "h100", "a40", "h200", "anybig"],
        default="anybig",
        help="Type of GPU to request (default: anybig)",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to request (default: 1)"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="mimic-mjx",
        help="Name of the SLURM job (default: mimic-mjx)",
    )
    parser.add_argument(
        "--mem", type=int, default=64000, help="Memory in MB (default: 64000)"
    )
    parser.add_argument(
        "--cpus", type=int, default=4, help="Number of CPU cores (default: 4)"
    )
    parser.add_argument(
        "--time",
        type=str,
        default="1-00:00",
        help="Time limit for the job (default: 1-0:00)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="slurm-out",
        help="Path for standard output (default: slurm-out)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="tracking",
        help="name of training task; tracking, bowl_escape_transfer, highlvl, task_ppo, "
        "task_ppo_warp (alias for task_ppo with Warp), bowl_escape_highlvl, "
        "rodent_rear_highlvl, sparse_imitation_highlvl, imitation_highlvl "
        "(default: tracking). For highlvl/task_ppo/task_ppo_warp, use -- "
        "separator to pass args to the script",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="charles_1gpu_v1",
        help="Name of the Hydra config to use (default: charles_1gpu_v1)",
    )

    args, hydra_overrides = parser.parse_known_args()

    submit(
        task=args.task,
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        out_dir=args.out_dir,
        config_name=args.config_name,
        hydra_overrides=hydra_overrides,
    )


if __name__ == "__main__":
    main()
