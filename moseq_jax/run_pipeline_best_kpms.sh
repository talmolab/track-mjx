#!/bin/bash
# =============================================================================
# Best-KPMS Pipeline: KPMS Sweep → Code Generation → Decoder Sweep
# =============================================================================
#
# Stage 1: KPMS sweep (num_states=25, sweep kappa) → select by MSE
#           Logs fitted vs actual signals to WandB
# Stage 2: Generate ONE code database from best model
# Stage 3: Decoder RL sweep over (kl_weight × kl_schedule × dropout)
#
# Usage:
#   cd moseq_jax
#   bash run_pipeline_best_kpms.sh
#   bash run_pipeline_best_kpms.sh --skip-kpms       # reuse existing codes
#   bash run_pipeline_best_kpms.sh --dry-run
#   bash run_pipeline_best_kpms.sh --config configs/custom.yaml
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

CONFIG="$SCRIPT_DIR/configs/pipeline_best_kpms.yaml"
SKIP_KPMS=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2"; shift 2 ;;
        --skip-kpms)  SKIP_KPMS=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash run_pipeline_best_kpms.sh [--config FILE] [--skip-kpms] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Config reader
# ---------------------------------------------------------------------------
read_yaml() {
    $VENV_PYTHON -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
v = cfg
for k in keys:
    v = v[k]
if isinstance(v, list):
    print(' '.join(str(x) for x in v))
else:
    print(v)
"
}

# ---------------------------------------------------------------------------
# Read config
# ---------------------------------------------------------------------------
OUTPUT_DIR=$(read_yaml output.base_dir)
BALANCED_SPLIT=$(read_yaml data.balanced_split_path)

# Decoder grid
KL_WEIGHTS=$(read_yaml decoder.kl_weights)
KL_SCHEDULES=$(read_yaml decoder.kl_schedules)
DROPOUT_RATES=$(read_yaml decoder.z_e_dropout_rates)
CODE_STACK_SIZE=$(read_yaml decoder.code_stack_size)
LATENT_DIM=$(read_yaml decoder.continuous_latent_dim)
USE_RNN=$(read_yaml decoder.use_rnn_decoder)
RNN_HIDDEN=$(read_yaml decoder.rnn_hidden_sizes)
RNN_CELL=$(read_yaml decoder.rnn_cell_type)
NUM_TIMESTEPS=$(read_yaml decoder.num_timesteps)
EVAL_EVERY=$(read_yaml decoder.eval_every)

WANDB_PROJECT=$(read_yaml wandb.project)
WANDB_GROUP=$(read_yaml wandb.group)

echo "=============================================="
echo "  Best-KPMS Pipeline"
echo "=============================================="
echo "Config:          $CONFIG"
echo "Output:          $OUTPUT_DIR"
echo "KL weights:      $KL_WEIGHTS"
echo "KL schedules:    $KL_SCHEDULES"
echo "Dropout rates:   $DROPOUT_RATES"
echo "Code stack:      $CODE_STACK_SIZE"
echo "Skip KPMS:       $SKIP_KPMS"
echo "Dry run:         $DRY_RUN"
echo "=============================================="

# ---------------------------------------------------------------------------
# Stage 1: KPMS Sweep
# ---------------------------------------------------------------------------
BEST_CODES="$SCRIPT_DIR/$OUTPUT_DIR/best_codes.npz"
SWEEP_RESULTS="$SCRIPT_DIR/$OUTPUT_DIR/sweep_results.json"

if [ "$SKIP_KPMS" = false ]; then
    echo ""
    echo "=== Stage 1: KPMS Sweep (jax_enable_x64=True) ==="
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run KPMS sweep with config: $CONFIG"
    else
        cd "$SCRIPT_DIR"
        # run_sweep.py reads 'sweep' section from the config
        # It logs fitted vs actual signals to WandB
        $VENV_PYTHON -m sweep.run_sweep --config "$CONFIG"
        echo "KPMS sweep complete."
    fi

    # Stage 2: Generate codes from best model
    echo ""
    echo "=== Stage 2: Generate Codes from Best Model ==="
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would generate codes from best KPMS model"
    else
        cd "$SCRIPT_DIR"
        $VENV_PYTHON -m codegen.generate_codes \
            --sweep-results "$SWEEP_RESULTS" \
            --balanced-split "$BALANCED_SPLIT" \
            --output-dir "$SCRIPT_DIR/$OUTPUT_DIR"
        echo "Code generation complete: $BEST_CODES"
    fi
else
    echo ""
    echo "=== Stages 1+2: SKIPPED (--skip-kpms) ==="
    if [ "$DRY_RUN" = false ] && [ ! -f "$BEST_CODES" ]; then
        echo "ERROR: $BEST_CODES not found. Run without --skip-kpms first."
        exit 1
    fi
fi

# Print best model info
if [ "$DRY_RUN" = false ] && [ -f "$SWEEP_RESULTS" ]; then
    echo ""
    $VENV_PYTHON -c "
import json
with open('$SWEEP_RESULTS') as f:
    r = json.load(f)
b = r['best_model']
print(f'Best KPMS model:')
print(f'  kappa:      {b[\"kappa\"]}')
print(f'  n_states:   {b[\"n_states\"]}')
print(f'  MSE:        {b[\"reconstruction_mse\"]:.6f}')
print(f'  duration:   {b[\"mean_duration\"]:.1f} frames')
print(f'  usage:      {b[\"syllable_usage_ratio\"]:.2%}')
"
fi

# ---------------------------------------------------------------------------
# Stage 3: Decoder Sweep
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3: Decoder RL Sweep ==="

# Get num_codes from codes file
if [ "$DRY_RUN" = false ]; then
    NUM_CODES=$($VENV_PYTHON -c "
import numpy as np
d = np.load('$BEST_CODES')
print(int(max(d['train_codes'].max(), d['test_codes'].max())) + 1)
")
else
    NUM_CODES="<auto>"
fi

# Count total runs
TOTAL_RUNS=0
for kl_w in $KL_WEIGHTS; do
    for kl_s in $KL_SCHEDULES; do
        for dropout in $DROPOUT_RATES; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done
done
echo "Total decoder runs: $TOTAL_RUNS"
echo ""

COMPLETED=0
FAILED=0
FAILED_RUNS=""
RUN_IDX=0

for kl_w in $KL_WEIGHTS; do
    for kl_s in $KL_SCHEDULES; do
        for dropout in $DROPOUT_RATES; do
            RUN_IDX=$((RUN_IDX + 1))
            RUN_NAME="kl${kl_w}_${kl_s}_drop${dropout}_stack${CODE_STACK_SIZE}"

            echo "--- [$RUN_IDX/$TOTAL_RUNS] $RUN_NAME ---"

            # Build KL schedule overrides
            KL_SCHED_OVERRIDES=""
            if [ "$kl_s" = "ramp" ]; then
                RAMP_START=$($VENV_PYTHON -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['kl_schedule_configs']['ramp']['start_frac'])")
                RAMP_END=$($VENV_PYTHON -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['kl_schedule_configs']['ramp']['end_frac'])")
                RAMP_START_VAL=$($VENV_PYTHON -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['kl_schedule_configs']['ramp']['start_value'])")
                KL_SCHED_OVERRIDES="network_config.kl_schedule_config.start_value=$RAMP_START_VAL network_config.kl_schedule_config.end_value=$kl_w network_config.kl_schedule_config.start_frac=$RAMP_START network_config.kl_schedule_config.end_frac=$RAMP_END"
            elif [ "$kl_s" = "cosine_anneal" ]; then
                COS_START=$($VENV_PYTHON -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['kl_schedule_configs']['cosine_anneal']['start_frac'])")
                COS_CYCLES=$($VENV_PYTHON -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['kl_schedule_configs']['cosine_anneal']['num_cycles'])")
                COS_START_VAL=$($VENV_PYTHON -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['kl_schedule_configs']['cosine_anneal']['start_value'])")
                KL_SCHED_OVERRIDES="network_config.kl_schedule_config.start_value=$COS_START_VAL network_config.kl_schedule_config.end_value=$kl_w network_config.kl_schedule_config.start_frac=$COS_START network_config.kl_schedule_config.num_cycles=$COS_CYCLES"
            fi

            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] kl=$kl_w, sched=$kl_s, drop=$dropout, stack=$CODE_STACK_SIZE"
                COMPLETED=$((COMPLETED + 1))
                continue
            fi

            cd "$SCRIPT_DIR"
            if $VENV_PYTHON train_moseq_decoder.py \
                kpms_config.codes_path="$BEST_CODES" \
                network_config.num_codes="$NUM_CODES" \
                network_config.kl_weight="$kl_w" \
                network_config.kl_schedule="$kl_s" \
                $KL_SCHED_OVERRIDES \
                network_config.z_e_dropout_rate="$dropout" \
                network_config.code_stack_size="$CODE_STACK_SIZE" \
                network_config.continuous_latent_dim="$LATENT_DIM" \
                network_config.use_continuous_encoder=true \
                network_config.use_rnn_decoder="$USE_RNN" \
                "network_config.rnn_hidden_sizes=[$RNN_HIDDEN]" \
                network_config.rnn_cell_type="$RNN_CELL" \
                network_config.z_e_at_action_head=true \
                network_config.reinit_hidden_on_code=true \
                network_config.learned_hidden_init=true \
                train_setup.run_name="$RUN_NAME" \
                train_setup.train_config.num_timesteps="$NUM_TIMESTEPS" \
                train_setup.eval_every="$EVAL_EVERY" \
                logging_config.group_name="$WANDB_GROUP" \
                logging_config.exp_name="$RUN_NAME"; then
                COMPLETED=$((COMPLETED + 1))
                echo "  DONE ($COMPLETED/$TOTAL_RUNS)"
            else
                FAILED=$((FAILED + 1))
                FAILED_RUNS="$FAILED_RUNS $RUN_NAME"
                echo "  FAILED ($FAILED failures)"
            fi
            echo ""
        done
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=============================================="
echo "  Pipeline Complete"
echo "=============================================="
echo "Total:     $TOTAL_RUNS"
echo "Completed: $COMPLETED"
echo "Failed:    $FAILED"
if [ -n "$FAILED_RUNS" ]; then
    echo "Failed:   $FAILED_RUNS"
fi
echo "=============================================="
