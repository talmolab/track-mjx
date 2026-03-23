"""Generate KPMS syllable codes for every setting in a sweep.

Instead of extracting codes only from the single global best model (as
``generate_codes.py`` does), this script iterates over every unique
``(num_states, kappa)`` combination in the sweep results, picks the best
seed for that setting, and writes a separate ``.npz`` file.

A ``manifest.json`` index is produced so downstream scripts (the decoder
trainer) know which code files exist and their metadata.

**CRITICAL**: Sets ``jax_enable_x64 = True`` — must run in a separate
process from RL training.

Usage::

    cd moseq_jax
    python -m codegen.generate_all_codes \\
        --sweep-results outputs/pipeline_sweep/kpms/sweep_results.json \\
        --balanced-split ../data/rodent/rodent_balanced_splits.json \\
        --output-dir outputs/pipeline_sweep/codes
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
sys.path.insert(0, str(MOSEQ_DIR))
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def generate_all_codes(
    sweep_results_path: str,
    balanced_split_path: str,
    output_dir: str,
) -> dict:
    """Generate per-setting code files from a KPMS sweep.

    For each unique ``(num_states, kappa)`` setting, selects the best seed
    (lowest MSE, then highest EML, then highest usage ratio) and extracts
    syllable codes aligned to the balanced train/test split.

    Args:
        sweep_results_path: Path to ``sweep_results.json`` from the KPMS sweep.
        balanced_split_path: Path to the balanced split JSON.
        output_dir: Directory to write per-setting ``.npz`` files and manifest.

    Returns:
        Manifest dict mapping setting names to metadata.
    """
    import h5py

    with open(sweep_results_path) as f:
        data = json.load(f)

    all_results = [r for r in data["all_results"] if "error" not in r]
    if not all_results:
        raise ValueError("No successful fits in sweep results")

    # Group by (num_states, kappa)
    settings: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["n_states"], r["kappa"])
        settings[key].append(r)

    log.info(f"Found {len(settings)} unique (num_states, kappa) settings")

    # Load balanced split indices
    with open(balanced_split_path) as f:
        splits = json.load(f)
    train_indices = splits["balanced"]["train_indices"]
    test_indices = splits["balanced"]["test_indices"]
    all_indices = sorted(set(train_indices) | set(test_indices))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve sweep results dir for relative project_dir paths
    sweep_base = Path(sweep_results_path).resolve().parent.parent

    manifest: dict[str, dict] = {}

    for (n_states, kappa), results in sorted(settings.items()):
        # Pick best seed for this setting
        results.sort(
            key=lambda r: (
                r["reconstruction_mse"],
                -r.get("eml_score", float("-inf")),
                -r["syllable_usage_ratio"],
            )
        )
        best = results[0]
        setting_name = f"s{n_states}_k{kappa:.0e}_arhmm"

        log.info(
            f"  {setting_name}: seed={best['seed']}, "
            f"MSE={best['reconstruction_mse']:.6f}, "
            f"EML={best.get('eml_score', 'N/A')}, "
            f"dur={best['mean_duration']:.1f}"
        )

        # Resolve project_dir (may be relative to sweep CWD = moseq_jax/)
        project_dir = best["project_dir"]
        results_h5 = Path(project_dir) / best["model_name"] / "results.h5"
        if not results_h5.exists():
            # Try relative to moseq_jax/
            results_h5 = MOSEQ_DIR / project_dir / best["model_name"] / "results.h5"
        if not results_h5.exists():
            log.warning(f"  results.h5 not found for {setting_name}, skipping")
            continue

        with h5py.File(str(results_h5), "r") as f:
            rec_names = sorted(f.keys())
            all_labels = [f[rn]["syllable"][:] for rn in rec_names]

        n_balanced = len(all_indices)
        if len(all_labels) != n_balanced:
            log.warning(
                f"  Expected {n_balanced} recordings, got {len(all_labels)}, skipping"
            )
            continue

        all_codes = np.array(all_labels, dtype=np.int32)

        # Map balanced indices to train/test positions
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
        train_positions = [idx_to_pos[idx] for idx in train_indices]
        test_positions = [idx_to_pos[idx] for idx in test_indices]

        train_codes = all_codes[train_positions]
        test_codes = all_codes[test_positions]

        num_codes = int(np.max(all_codes)) + 1

        # Save
        output_path = out / f"{setting_name}.npz"
        np.savez(
            str(output_path),
            all_codes=all_codes,
            train_codes=train_codes,
            test_codes=test_codes,
            train_indices=np.array(train_indices),
            test_indices=np.array(test_indices),
        )

        manifest[setting_name] = {
            "codes_path": str(output_path.resolve()),
            "num_codes": num_codes,
            "n_states": n_states,
            "kappa": kappa,
            "seed": best["seed"],
            "model_name": best["model_name"],
            "mean_duration": best["mean_duration"],
            "transition_entropy": best["transition_entropy"],
        }

        log.info(
            f"    -> {output_path.name}: "
            f"shape={all_codes.shape}, num_codes={num_codes}"
        )

    # Write manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Wrote manifest ({len(manifest)} settings) to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-setting KPMS codes from a sweep"
    )
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
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-setting .npz files",
    )
    args = parser.parse_args()

    generate_all_codes(
        sweep_results_path=args.sweep_results,
        balanced_split_path=args.balanced_split,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
