"""Select K clips per behaviour category from balanced splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np


def load_balanced_splits(path: str | Path) -> dict:
    """Load the balanced-split JSON and return the full dict."""
    with open(path) as f:
        return json.load(f)


def select_clips_by_behavior(
    splits: dict,
    split: str,
    k_per_behavior: int,
    seed: int = 42,
    behaviors: tuple[str, ...] = ("groom", "walk", "rear"),
) -> dict[str, list[int]]:
    """Pick *k_per_behavior* clip indices for each behaviour.

    Args:
        splits: The full balanced-split dict (from ``load_balanced_splits``).
        split: ``"train"`` or ``"test"``.
        k_per_behavior: How many clips to select per category.
        seed: Random seed for reproducible selection.
        behaviors: Which categories to pick from.

    Returns:
        ``{behavior: [local_clip_indices]}`` where indices are positions
        within the split (0-based into train_codes / test_codes).
    """
    indices = splits["balanced"][f"{split}_indices"]
    categories = splits["balanced"][f"{split}_categories"]

    rng = np.random.RandomState(seed)
    result: dict[str, list[int]] = {}

    for beh in behaviors:
        # Local indices (position within the split) that match this category
        candidates = [i for i, c in enumerate(categories) if c == beh]
        if len(candidates) == 0:
            # Fall back to rear_walk if "rear" not present
            if beh == "rear":
                candidates = [
                    i for i, c in enumerate(categories) if c == "rear_walk"
                ]
        if len(candidates) < k_per_behavior:
            logging.warning(
                f"Only {len(candidates)} clips for '{beh}' in {split} split "
                f"(requested {k_per_behavior})"
            )
            selected = candidates
        else:
            selected = rng.choice(candidates, size=k_per_behavior, replace=False).tolist()
        result[beh] = selected
        logging.info(
            f"  {beh}: selected {len(selected)} clips "
            f"(global indices {[indices[s] for s in selected]})"
        )

    return result
