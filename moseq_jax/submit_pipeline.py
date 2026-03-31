"""Submit the MoSeq pipeline as SLURM jobs with dependency chaining.

Submits 3 stages as separate SLURM jobs:
  Stage 1: KPMS sweep (jax_enable_x64, single GPU job)
  Stage 2: Code generation + decoder grid file (jax_enable_x64, single GPU job)
  Stage 3: Decoder RL training (array job, one task per HP combination)

Usage:
    cd track-mjx
    python moseq_jax/submit_pipeline.py                          # full pipeline
    python moseq_jax/submit_pipeline.py --skip-kpms              # reuse sweep
    python moseq_jax/submit_pipeline.py --skip-kpms --skip-codegen  # only decoders
    python moseq_jax/submit_pipeline.py --dry-run                # show scripts
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

MOSEQ_DIR = Path(__file__).resolve().parent
REPO_ROOT = MOSEQ_DIR.parent

# The .venv lives in the main repo, not in worktrees
_git_common = subprocess.check_output(
    ["git", "rev-parse", "--git-common-dir"],
    cwd=str(REPO_ROOT),
    text=True,
).strip()
MAIN_REPO = Path(_git_common).resolve().parent
VENV_DIR = MAIN_REPO / ".venv"
KPMS_VENV_DIR = MAIN_REPO / ".venv-kpms"  # separate venv for keypoint-moseq (jax<0.7)


def _slurm_submit(script: str, dry_run: bool = False) -> str | None:
    """Submit a SLURM script via sbatch stdin. Returns job ID or None."""
    if dry_run:
        print(script)
        print("=" * 60)
        return None
    try:
        output = subprocess.check_output(
            ["sbatch"], input=script, universal_newlines=True
        )
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def _env_setup(venv: Path = VENV_DIR) -> str:
    """Shared SLURM environment setup block."""
    return f"""
# Load modules and activate environment
source ~/.bashrc
module load python
module load cuda/12.4.1-fasrc01
cd {REPO_ROOT}
source {venv}/bin/activate

# Display GPU information
nvidia-smi

# Environment variables
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export WANDB_CACHE_DIR=$PWD
export PYTHONUNBUFFERED=1

# Add pip nvidia lib paths for JAX CUDA
NVIDIA_LIBS=$(python3 -c "import nvidia; from pathlib import Path; print(':'.join(str(p) for p in Path(nvidia.__path__[0]).glob('*/lib') if p.is_dir()))" 2>/dev/null)
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
"""


def _sbatch_header(
    job_name: str,
    time: str,
    gpu: bool = True,
    gpu_type: str = "anybig",
    mem: int = 64000,
    cpus: int = 4,
    dependency: str | None = None,
    array: str | None = None,
    partition: str = "kempner,kempner_h100",
    account: str = "kempner_pehlevan_lab",
) -> str:
    """Build #SBATCH header lines."""
    gpu_configs = {
        "a100": "nvidia_a100-sxm4-40gb",
        "h100": "nvidia_h100_80gb_hbm3",
        "h200": "nvidia_h200",
    }

    lines = [
        "#!/bin/bash",
        f"#SBATCH -p {partition}",
        f"#SBATCH -A {account}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -c {cpus}",
        "#SBATCH -N 1",
        f"#SBATCH -t {time}",
        f"#SBATCH -J {job_name}",
    ]

    if gpu:
        if gpu_type in gpu_configs:
            lines.append(f"#SBATCH --gres=gpu:{gpu_configs[gpu_type]}:1")
        else:
            lines.append("#SBATCH --gres=gpu:1")
            lines.append('#SBATCH --constraint="a100|h200|h100"')

    lines.append(f"#SBATCH -o {MOSEQ_DIR}/slurm-out/%x_%j.out")
    lines.append(f"#SBATCH -e {MOSEQ_DIR}/slurm-out/%x_%j.err")

    if dependency:
        lines.append(f"#SBATCH --dependency=afterok:{dependency}")

    if array:
        lines.append(f"#SBATCH --array={array}")

    return "\n".join(lines)


def _build_stage1_script(config_path: str, gpu_type: str, time: str, partition: str = "kempner,kempner_h100", account: str = "kempner_pehlevan_lab") -> str:
    """Stage 1: KPMS sweep."""
    header = _sbatch_header(
        job_name="moseq-kpms-sweep",
        time=time,
        gpu_type=gpu_type,
        partition=partition,
        account=account,
    )
    return f"""{header}
{_env_setup(KPMS_VENV_DIR)}
# Stage 1: KPMS sweep (requires jax_enable_x64, uses .venv-kpms)
export JAX_ENABLE_X64=1
cd {MOSEQ_DIR}
python -m sweep.run_sweep --config {config_path}
"""


def _build_stage2_script(
    config_path: str,
    gpu_type: str,
    dependency: str | None = None,
    partition: str = "kempner,kempner_h100",
    account: str = "kempner_pehlevan_lab",
) -> str:
    """Stage 2: Code generation + decoder grid file."""
    header = _sbatch_header(
        job_name="moseq-codegen",
        time="0-02:00",
        gpu_type=gpu_type,
        mem=32000,
        dependency=dependency,
        partition=partition,
        account=account,
    )

    return f"""{header}
{_env_setup(KPMS_VENV_DIR)}
# Stage 2: Per-setting code generation (requires jax_enable_x64, uses .venv-kpms)
export JAX_ENABLE_X64=1
cd {MOSEQ_DIR}

# Read paths from pipeline config
SWEEP_RESULTS=$(python3 -c "
import yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)
print(cfg['codegen']['sweep_results'])
")
BALANCED_SPLIT=$(python3 -c "
import yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)
print(cfg['codegen']['balanced_split'])
")
CODES_OUTPUT=$(python3 -c "
import yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)
print(cfg['codegen']['output_dir'])
")

python -m codegen.generate_all_codes \\
    --sweep-results "$SWEEP_RESULTS" \\
    --balanced-split "$BALANCED_SPLIT" \\
    --output-dir "$CODES_OUTPUT"

# Generate decoder grid mapping for Stage 3 array job
python3 -c "
import json, itertools, yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)
codes_dir = cfg['codegen']['output_dir']
manifest_path = codes_dir + '/manifest.json'
with open(manifest_path) as f:
    manifest = json.load(f)
latent_dims = cfg['decoder']['continuous_latent_dims']
kl_weights = cfg['decoder']['kl_weights']
grid = []
for idx, (setting, ld, kl) in enumerate(
    itertools.product(sorted(manifest.keys()), latent_dims, kl_weights)
):
    entry = manifest[setting].copy()
    entry.update({{'idx': idx, 'setting': setting, 'latent_dim': ld, 'kl_weight': kl}})
    grid.append(entry)
out_path = codes_dir + '/decoder_grid.json'
with open(out_path, 'w') as f:
    json.dump(grid, f, indent=2)
print(f'Wrote {{len(grid)}} decoder grid entries to {{out_path}}')
"
"""


def _build_stage3_script(
    config_path: str,
    grid_size: int,
    gpu_type: str,
    time: str,
    throttle: int,
    dependency: str | None = None,
    partition: str = "kempner,kempner_h100",
    account: str = "kempner_pehlevan_lab",
) -> str:
    """Stage 3: Decoder RL training array job."""
    array_spec = f"0-{grid_size - 1}%{throttle}"
    header = _sbatch_header(
        job_name="moseq-decoder",
        time=time,
        gpu_type=gpu_type,
        array=array_spec,
        dependency=dependency,
        partition=partition,
        account=account,
    )

    return f"""{header}
{_env_setup()}
# Stage 3: Decoder RL training (one run per array task)
cd {MOSEQ_DIR}

# Read pipeline config for decoder defaults
GRID_FILE=$(python3 -c "
import yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)
print(cfg['codegen']['output_dir'] + '/decoder_grid.json')
")

# Extract parameters for this array task
eval "$(python3 -c "
import json
with open('$GRID_FILE') as f:
    grid = json.load(f)
e = grid[$SLURM_ARRAY_TASK_ID]
print(f\\"CODES_PATH={{e['codes_path']}}\\")
print(f\\"NUM_CODES={{e['num_codes']}}\\")
print(f\\"LATENT_DIM={{e['latent_dim']}}\\")
print(f\\"KL_WEIGHT={{e['kl_weight']}}\\")
print(f\\"SETTING={{e['setting']}}\\")
")"

RUN_NAME="${{SETTING}}_ld${{LATENT_DIM}}_kl${{KL_WEIGHT}}"

# Read remaining decoder config from pipeline yaml
DECODER_CFG=$(python3 -c "
import yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)
d = cfg['decoder']
print(f\\"NUM_TIMESTEPS={{d['num_timesteps']}}\\")
print(f\\"EVAL_EVERY={{d['eval_every']}}\\")
print(f\\"USE_RNN={{str(d['use_rnn_decoder']).lower()}}\\")
print(f\\"RNN_HIDDEN={{','.join(str(x) for x in d['rnn_hidden_sizes'])}}\\")
print(f\\"RNN_CELL={{d['rnn_cell_type']}}\\")
print(f\\"USE_CONT_ENC={{str(d['use_continuous_encoder']).lower()}}\\")
print(f\\"Z_E_ANNEAL={{str(d['z_e_anneal']).lower()}}\\")
print(f\\"Z_E_ANNEAL_START={{d['z_e_anneal_start_frac']}}\\")
print(f\\"Z_E_ANNEAL_END={{d['z_e_anneal_end_frac']}}\\")
print(f\\"NUM_ENVS={{d.get('num_envs', 2048)}}\\")
w = cfg.get('wandb', {{}})
print(f\\"WANDB_GROUP={{w.get('group', 'pipeline_sweep')}}\\")
")
eval "$DECODER_CFG"

echo "=== Task $SLURM_ARRAY_TASK_ID: $RUN_NAME ==="
echo "  codes: $CODES_PATH (num_codes=$NUM_CODES)"
echo "  latent_dim=$LATENT_DIM, kl_weight=$KL_WEIGHT"
echo "  num_envs=$NUM_ENVS, num_timesteps=$NUM_TIMESTEPS"

python train_moseq_decoder.py \\
    kpms_config.codes_path="$CODES_PATH" \\
    network_config.num_codes="$NUM_CODES" \\
    network_config.continuous_latent_dim="$LATENT_DIM" \\
    network_config.kl_weight="$KL_WEIGHT" \\
    network_config.use_continuous_encoder="$USE_CONT_ENC" \\
    network_config.use_rnn_decoder="$USE_RNN" \\
    "network_config.rnn_hidden_sizes=[$RNN_HIDDEN]" \\
    network_config.rnn_cell_type="$RNN_CELL" \\
    network_config.z_e_anneal="$Z_E_ANNEAL" \\
    network_config.z_e_anneal_start_frac="$Z_E_ANNEAL_START" \\
    network_config.z_e_anneal_end_frac="$Z_E_ANNEAL_END" \\
    train_setup.run_name="$RUN_NAME" \\
    train_setup.train_config.num_timesteps="$NUM_TIMESTEPS" \\
    train_setup.train_config.num_envs="$NUM_ENVS" \\
    train_setup.eval_every="$EVAL_EVERY" \\
    logging_config.group_name="$WANDB_GROUP" \\
    logging_config.exp_name="$RUN_NAME"
"""


def main():
    parser = argparse.ArgumentParser(
        description="Submit MoSeq pipeline as SLURM jobs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(MOSEQ_DIR / "configs" / "pipeline_sweep.yaml"),
        help="Pipeline config YAML (default: configs/pipeline_sweep.yaml)",
    )
    parser.add_argument("--skip-kpms", action="store_true", help="Skip KPMS sweep")
    parser.add_argument("--skip-codegen", action="store_true", help="Skip codegen")
    parser.add_argument("--dry-run", action="store_true", help="Print scripts only")
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="anybig",
        choices=["a100", "h100", "h200", "anybig"],
        help="GPU type (default: anybig)",
    )
    parser.add_argument(
        "--time-kpms",
        type=str,
        default="0-04:00",
        help="Walltime for KPMS sweep (default: 4h)",
    )
    parser.add_argument(
        "--time-decoder",
        type=str,
        default="1-00:00",
        help="Walltime per decoder run (default: 1d)",
    )
    parser.add_argument(
        "--throttle",
        type=int,
        default=10,
        help="Max concurrent decoder array tasks (default: 10)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="kempner,kempner_h100",
        help="SLURM partition (default: kempner,kempner_h100)",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="kempner_pehlevan_lab",
        help="SLURM account (default: kempner_pehlevan_lab)",
    )
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())

    # Load config to compute grid size
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_settings = len(cfg["sweep"]["num_states"]) * len(cfg["sweep"]["kappa"])
    n_latent = len(cfg["decoder"]["continuous_latent_dims"])
    n_kl = len(cfg["decoder"]["kl_weights"])
    grid_size = n_settings * n_latent * n_kl

    # Ensure slurm output dir exists
    slurm_out = MOSEQ_DIR / "slurm-out"
    slurm_out.mkdir(exist_ok=True)

    # Check prerequisites
    balanced_split = cfg["data"]["balanced_split_path"]
    if not Path(balanced_split).exists():
        print(f"ERROR: balanced_split_path not found: {balanced_split}")
        print("Run: python vqvae_jax/preprocess_clips.py --data_path data/rodent/rodent_reference_clips.h5 --output_path data/rodent/rodent_balanced_splits.json --no_video")
        sys.exit(1)

    ref_data = cfg["data"]["reference_data_path"]
    if not Path(ref_data).exists():
        print(f"ERROR: reference_data_path not found: {ref_data}")
        sys.exit(1)

    print("=" * 60)
    print("  MoSeq Pipeline SLURM Submission")
    print("=" * 60)
    print(f"Config:          {config_path}")
    print(f"Grid size:       {n_settings} KPMS settings x {n_latent} latent_dims x {n_kl} kl_weights = {grid_size} decoder runs")
    print(f"GPU type:        {args.gpu_type}")
    print(f"Skip KPMS:       {args.skip_kpms}")
    print(f"Skip codegen:    {args.skip_codegen}")
    print(f"Decoder time:    {args.time_decoder}")
    print(f"Throttle:        {args.throttle}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 60)

    stage1_id = None
    stage2_id = None

    # Stage 1: KPMS sweep
    if not args.skip_kpms:
        print("\n--- Stage 1: KPMS Sweep ---")
        script = _build_stage1_script(config_path, args.gpu_type, args.time_kpms, args.partition, args.account)
        stage1_id = _slurm_submit(script, args.dry_run)
        if stage1_id:
            print(f"Submitted: job {stage1_id}")
    else:
        print("\n--- Stage 1: SKIPPED ---")
        sweep_results = MOSEQ_DIR / cfg["codegen"]["sweep_results"]
        if not sweep_results.exists():
            print(f"WARNING: sweep_results.json not found at {sweep_results}")

    # Stage 2: Codegen
    if not args.skip_codegen:
        print("\n--- Stage 2: Code Generation ---")
        script = _build_stage2_script(config_path, args.gpu_type, stage1_id, args.partition, args.account)
        stage2_id = _slurm_submit(script, args.dry_run)
        if stage2_id:
            print(f"Submitted: job {stage2_id}")
    else:
        print("\n--- Stage 2: SKIPPED ---")
        grid_file = MOSEQ_DIR / cfg["codegen"]["output_dir"] / "decoder_grid.json"
        if not grid_file.exists():
            print(f"WARNING: decoder_grid.json not found at {grid_file}")

    # Stage 3: Decoder training array
    print(f"\n--- Stage 3: Decoder Training ({grid_size} runs) ---")
    script = _build_stage3_script(
        config_path, grid_size, args.gpu_type, args.time_decoder, args.throttle, stage2_id,
        args.partition, args.account,
    )
    stage3_id = _slurm_submit(script, args.dry_run)
    if stage3_id:
        print(f"Submitted: array job {stage3_id}")

    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN complete — no jobs submitted")
    else:
        print("Pipeline submitted!")
        ids = []
        if stage1_id:
            ids.append(f"Stage 1: {stage1_id}")
        if stage2_id:
            ids.append(f"Stage 2: {stage2_id}")
        if stage3_id:
            ids.append(f"Stage 3: {stage3_id}")
        for line in ids:
            print(f"  {line}")
        print(f"\nMonitor: squeue -u $USER")
    print("=" * 60)


if __name__ == "__main__":
    main()
