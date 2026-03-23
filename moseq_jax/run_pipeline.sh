#!/bin/bash
# =============================================================================
# MoSeq Full Pipeline: KPMS Sweep -> Code Generation -> Decoder Training
# =============================================================================
#
# Runs the complete hyperparameter tuning pipeline in 3 stages:
#
#   Stage 1  KPMS sweep         (jax_enable_x64, separate process)
#   Stage 2  Per-setting codegen (jax_enable_x64, separate process)
#   Stage 3  Decoder RL training (float32, one run per combination)
#
# Usage:
#   cd moseq_jax
#   bash run_pipeline.sh                             # full pipeline
#   bash run_pipeline.sh --skip-kpms                 # reuse existing sweep
#   bash run_pipeline.sh --skip-kpms --skip-codegen  # only run decoders
#   bash run_pipeline.sh --config configs/custom.yaml
#   bash run_pipeline.sh --dry-run                   # print plan, don't run
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

# ---------------------------------------------------------------------------
# Defaults (overridden by pipeline_sweep.yaml)
# ---------------------------------------------------------------------------
CONFIG="$SCRIPT_DIR/configs/pipeline_sweep.yaml"
SKIP_KPMS=false
SKIP_CODEGEN=false
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --skip-kpms)   SKIP_KPMS=true; shift ;;
        --skip-codegen) SKIP_CODEGEN=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash run_pipeline.sh [--config FILE] [--skip-kpms] [--skip-codegen] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Read config with a small Python helper (avoids yq dependency)
# ---------------------------------------------------------------------------
read_yaml() {
    # Usage: read_yaml key.subkey config.yaml
    $VENV_PYTHON -c "
import yaml, sys
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
# Read pipeline parameters from config
# ---------------------------------------------------------------------------
SWEEP_OUTPUT=$(read_yaml output.base_dir)
CODES_OUTPUT=$(read_yaml codegen.output_dir)
SWEEP_RESULTS="$SWEEP_OUTPUT/sweep_results.json"
BALANCED_SPLIT=$(read_yaml codegen.balanced_split)
REFERENCE_DATA=$(read_yaml data.reference_data_path)
STAC_XML=$(read_yaml data.stac_xml_path)

# Decoder grid
LATENT_DIMS=$(read_yaml decoder.continuous_latent_dims)
KL_WEIGHTS=$(read_yaml decoder.kl_weights)
NUM_TIMESTEPS=$(read_yaml decoder.num_timesteps)
EVAL_EVERY=$(read_yaml decoder.eval_every)

# WandB
WANDB_PROJECT=$(read_yaml wandb.project)
WANDB_GROUP=$(read_yaml wandb.group)

echo "=============================================="
echo "  MoSeq Pipeline Sweep"
echo "=============================================="
echo "Config:          $CONFIG"
echo "Sweep output:    $SWEEP_OUTPUT"
echo "Codes output:    $CODES_OUTPUT"
echo "Latent dims:     $LATENT_DIMS"
echo "KL weights:      $KL_WEIGHTS"
echo "Timesteps/run:   $NUM_TIMESTEPS"
echo "Skip KPMS:       $SKIP_KPMS"
echo "Skip codegen:    $SKIP_CODEGEN"
echo "Dry run:         $DRY_RUN"
echo "=============================================="

# ---------------------------------------------------------------------------
# Stage 1: KPMS Sweep
# ---------------------------------------------------------------------------
if [ "$SKIP_KPMS" = false ]; then
    echo ""
    echo "=== Stage 1: KPMS Sweep ==="
    echo "Running KPMS hyperparameter sweep (jax_enable_x64=True)..."
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: $VENV_PYTHON -m sweep.run_sweep --config $CONFIG"
    else
        cd "$SCRIPT_DIR"
        $VENV_PYTHON -m sweep.run_sweep --config "$CONFIG"
        echo "KPMS sweep complete. Results: $SWEEP_RESULTS"
    fi
else
    echo ""
    echo "=== Stage 1: SKIPPED (--skip-kpms) ==="
    if [ "$DRY_RUN" = false ] && [ ! -f "$SCRIPT_DIR/$SWEEP_RESULTS" ] && [ ! -f "$SWEEP_RESULTS" ]; then
        echo "ERROR: sweep_results.json not found at $SWEEP_RESULTS"
        echo "Run without --skip-kpms first."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2: Per-setting code generation
# ---------------------------------------------------------------------------
if [ "$SKIP_CODEGEN" = false ]; then
    echo ""
    echo "=== Stage 2: Per-Setting Code Generation ==="
    echo ""

    # Resolve sweep results path (may be relative to moseq_jax/)
    if [ -f "$SCRIPT_DIR/$SWEEP_RESULTS" ]; then
        RESOLVED_SWEEP_RESULTS="$SCRIPT_DIR/$SWEEP_RESULTS"
    elif [ -f "$SWEEP_RESULTS" ]; then
        RESOLVED_SWEEP_RESULTS="$SWEEP_RESULTS"
    else
        if [ "$DRY_RUN" = true ]; then
            RESOLVED_SWEEP_RESULTS="$SCRIPT_DIR/$SWEEP_RESULTS"
        else
            echo "ERROR: Cannot find $SWEEP_RESULTS"
            exit 1
        fi
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: $VENV_PYTHON -m codegen.generate_all_codes \\"
        echo "    --sweep-results $RESOLVED_SWEEP_RESULTS \\"
        echo "    --balanced-split $BALANCED_SPLIT \\"
        echo "    --output-dir $CODES_OUTPUT"
    else
        cd "$SCRIPT_DIR"
        $VENV_PYTHON -m codegen.generate_all_codes \
            --sweep-results "$RESOLVED_SWEEP_RESULTS" \
            --balanced-split "$BALANCED_SPLIT" \
            --output-dir "$CODES_OUTPUT"
        echo "Code generation complete. Codes in: $CODES_OUTPUT"
    fi
else
    echo ""
    echo "=== Stage 2: SKIPPED (--skip-codegen) ==="
fi

# ---------------------------------------------------------------------------
# Stage 3: Decoder Training
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3: Decoder RL Training ==="

# Read manifest to get code file paths
MANIFEST="$SCRIPT_DIR/$CODES_OUTPUT/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    MANIFEST="$CODES_OUTPUT/manifest.json"
fi

if [ ! -f "$MANIFEST" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] No manifest.json yet — showing planned decoder grid:"
        echo ""
        # Generate synthetic setting names from config
        SETTINGS=$($VENV_PYTHON -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for ns in cfg['sweep']['num_states']:
    for k in cfg['sweep']['kappa']:
        print(f's{ns}_k{k:.0e}_arhmm')
")
    else
        echo "ERROR: manifest.json not found at $MANIFEST"
        echo "Run stages 1+2 first."
        exit 1
    fi
else
    # Extract setting names from manifest
    SETTINGS=$($VENV_PYTHON -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
for name in sorted(m.keys()):
    print(name)
")
fi

TOTAL_RUNS=0
COMPLETED=0
FAILED=0
FAILED_RUNS=""

# Count total runs
for setting in $SETTINGS; do
    for latent_dim in $LATENT_DIMS; do
        for kl_weight in $KL_WEIGHTS; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done
done

echo "Total decoder runs: $TOTAL_RUNS"
echo ""

RUN_IDX=0
for setting in $SETTINGS; do
    # Get codes_path and num_codes from manifest (or placeholders for dry-run)
    if [ -f "$MANIFEST" ]; then
        CODES_PATH=$($VENV_PYTHON -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
print(m['$setting']['codes_path'])
")
        NUM_CODES=$($VENV_PYTHON -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
print(m['$setting']['num_codes'])
")
    else
        CODES_PATH="<codes/${setting}.npz>"
        NUM_CODES="<auto>"
    fi

    for latent_dim in $LATENT_DIMS; do
        for kl_weight in $KL_WEIGHTS; do
            RUN_IDX=$((RUN_IDX + 1))
            RUN_NAME="${setting}_ld${latent_dim}_kl${kl_weight}"

            echo "--- [$RUN_IDX/$TOTAL_RUNS] $RUN_NAME ---"
            echo "  codes: $CODES_PATH (num_codes=$NUM_CODES)"
            echo "  latent_dim=$latent_dim, kl_weight=$kl_weight"

            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] Would run decoder training"
                COMPLETED=$((COMPLETED + 1))
                continue
            fi

            cd "$SCRIPT_DIR"
            if $VENV_PYTHON train_moseq_decoder.py \
                kpms_config.codes_path="$CODES_PATH" \
                network_config.num_codes="$NUM_CODES" \
                network_config.continuous_latent_dim="$latent_dim" \
                network_config.kl_weight="$kl_weight" \
                network_config.use_continuous_encoder=true \
                train_setup.run_name="$RUN_NAME" \
                train_setup.train_config.num_timesteps="$NUM_TIMESTEPS" \
                train_setup.eval_every="$EVAL_EVERY" \
                logging_config.group_name="$WANDB_GROUP" \
                logging_config.exp_name="$RUN_NAME"; then
                COMPLETED=$((COMPLETED + 1))
                echo "  DONE ($COMPLETED/$TOTAL_RUNS completed)"
            else
                FAILED=$((FAILED + 1))
                FAILED_RUNS="$FAILED_RUNS $RUN_NAME"
                echo "  FAILED ($FAILED failures so far)"
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
echo "Total runs:  $TOTAL_RUNS"
echo "Completed:   $COMPLETED"
echo "Failed:      $FAILED"
if [ -n "$FAILED_RUNS" ]; then
    echo "Failed runs:$FAILED_RUNS"
fi
echo "=============================================="
