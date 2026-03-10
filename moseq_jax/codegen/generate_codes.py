"""Generate KPMS syllable codes for all reference clips.

Loads the best KPMS model from a sweep and extracts syllable labels for each
clip, producing an ``.npz`` file with train/test code arrays aligned to the
balanced split indices.

**CRITICAL**: This script sets ``jax_enable_x64 = True`` and MUST run in a
separate process from the RL training.

Usage::

    cd moseq_jax
    python -m codegen.generate_codes \\
        --sweep-results moseq_jax/outputs/kpms_sweep/sweep_results.json \\
        --balanced-split data/rodent/rodent_balanced_splits.json \\
        --output moseq_jax/outputs/kpms_sweep/best_codes.npz
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
sys.path.insert(0, str(MOSEQ_DIR))
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def generate_codes(
    sweep_results_path: str,
    balanced_split_path: str,
    output_path: str,
) -> None:
    """Extract codes from the best KPMS model and save to ``.npz``.

    The output contains:
    - ``all_codes``: shape ``[N_balanced, T]`` int32
    - ``train_codes``: shape ``[N_train, T]`` int32
    - ``test_codes``: shape ``[N_test, T]`` int32
    - ``train_indices``: original clip indices for train set
    - ``test_indices``: original clip indices for test set

    Args:
        sweep_results_path: Path to ``sweep_results.json``.
        balanced_split_path: Path to balanced split JSON.
        output_path: Output ``.npz`` path.
    """
    import keypoint_moseq as kpms

    # Load sweep results
    with open(sweep_results_path) as f:
        results = json.load(f)

    best = results["best_model"]
    if best is None:
        raise ValueError("No best model found in sweep results")

    project_dir = best["project_dir"]
    model_name = best["model_name"]
    log.info(f"Best model: {model_name} at {project_dir}")

    # Load balanced split
    with open(balanced_split_path) as f:
        splits = json.load(f)
    train_indices = splits["balanced"]["train_indices"]
    test_indices = splits["balanced"]["test_indices"]
    all_indices = sorted(set(train_indices) | set(test_indices))

    # Load checkpoint and extract results
    model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
    config = kpms.load_config(project_dir)

    # The results dict has recording names sorted alphabetically
    results_dict = kpms.extract_results(
        model_ckpt,
        model_ckpt.get("metadata", None),
        project_dir,
        model_name,
    )

    rec_names = sorted(results_dict.keys())
    all_labels = [results_dict[rn]["syllable"] for rn in rec_names]

    # Map balanced indices to recording order
    # all_indices (sorted union of train+test) corresponds to the recording
    # names in sorted order
    n_balanced = len(all_indices)
    if len(all_labels) != n_balanced:
        raise ValueError(f"Expected {n_balanced} recordings but got {len(all_labels)}")

    all_codes = np.array(all_labels, dtype=np.int32)
    log.info(f"All codes shape: {all_codes.shape}")
    log.info(f"Unique codes: {np.unique(all_codes).tolist()}")

    # Build index mapping: all_indices[i] -> position i in all_codes
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    train_positions = [idx_to_pos[idx] for idx in train_indices]
    test_positions = [idx_to_pos[idx] for idx in test_indices]

    train_codes = all_codes[train_positions]
    test_codes = all_codes[test_positions]

    log.info(f"Train codes: {train_codes.shape}, Test codes: {test_codes.shape}")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        all_codes=all_codes,
        train_codes=train_codes,
        test_codes=test_codes,
        train_indices=np.array(train_indices),
        test_indices=np.array(test_indices),
    )
    log.info(f"Saved codes to {output_path}")

    # Print stats
    n_unique_train = len(np.unique(train_codes))
    n_unique_test = len(np.unique(test_codes))
    log.info(
        f"Train: {n_unique_train} unique codes, Test: {n_unique_test} unique codes"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate KPMS codes for RL training")
    parser.add_argument(
        "--sweep-results",
        type=str,
        required=True,
        help="Path to sweep_results.json",
    )
    parser.add_argument(
        "--balanced-split",
        type=str,
        required=True,
        help="Path to balanced split JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path",
    )
    args = parser.parse_args()

    generate_codes(args.sweep_results, args.balanced_split, args.output)


if __name__ == "__main__":
    main()
