"""Tests for rvq_analysis module: parent-child heatmap, diversity, transitions."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add vqvae_jax to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vqvae_jax"))

from analysis.inference_cache import InferenceResult
from analysis.rvq_analysis import (
    compute_parent_child_heatmap,
    compute_intra_parent_diversity,
    compute_hierarchical_transitions,
    run_rvq_analysis,
)


def _make_result(
    l0: list[int],
    l1: list[int],
    clip_idx: int = 0,
) -> InferenceResult:
    """Create a minimal InferenceResult with rvq_indices."""
    T = len(l0)
    return InferenceResult(
        clip_idx=clip_idx,
        code_indices=np.array(l0, dtype=np.int32),
        qpos=np.zeros((T, 10)),
        qvel=np.zeros((T, 10)),
        rewards=np.zeros(T),
        rvq_indices=(
            np.array(l0, dtype=np.int32),
            np.array(l1, dtype=np.int32),
        ),
    )


# =============================================================================
# 3a. Parent-Child Heatmap Tests
# =============================================================================


def test_parent_child_heatmap_basic():
    """Joint counts match hand-computed values."""
    # 3 frames: (L0=0,L1=1), (L0=0,L1=1), (L0=1,L1=0)
    result = _make_result([0, 0, 1], [1, 1, 0])
    fig, counts = compute_parent_child_heatmap([result], num_codes=4)

    assert counts.shape == (4, 4)
    assert counts[0, 1] == 2  # L0=0, L1=1 appears twice
    assert counts[1, 0] == 1  # L0=1, L1=0 appears once
    assert counts.sum() == 3
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_parent_child_heatmap_multiple_clips():
    """Joint counts accumulate across clips."""
    r1 = _make_result([0, 0], [1, 1], clip_idx=0)
    r2 = _make_result([0, 1], [1, 0], clip_idx=1)
    fig, counts = compute_parent_child_heatmap([r1, r2], num_codes=4)

    assert counts[0, 1] == 3  # 2 from r1 + 1 from r2
    assert counts[1, 0] == 1
    assert counts.sum() == 4
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_parent_child_heatmap_no_rvq():
    """Results without rvq_indices produce zero counts."""
    result = InferenceResult(
        clip_idx=0,
        code_indices=np.array([0, 1, 2]),
        qpos=np.zeros((3, 10)),
        qvel=np.zeros((3, 10)),
        rewards=np.zeros(3),
    )
    fig, counts = compute_parent_child_heatmap([result], num_codes=4)
    assert counts.sum() == 0
    import matplotlib.pyplot as plt

    plt.close(fig)


# =============================================================================
# 3b. Intra-Parent Diversity Tests
# =============================================================================


def test_intra_parent_diversity_uniform():
    """Uniform L1 distribution gives max entropy."""
    # Parent 0 maps equally to L1=0,1,2,3
    counts = np.zeros((4, 4), dtype=np.int64)
    counts[0, :] = [10, 10, 10, 10]
    counts[1, :] = [40, 0, 0, 0]  # Parent 1 always maps to L1=0

    fig, entropies = compute_intra_parent_diversity(counts)

    # Parent 0: uniform => entropy = log2(4) = 2.0
    np.testing.assert_almost_equal(entropies[0], 2.0, decimal=5)
    # Parent 1: deterministic => entropy = 0
    np.testing.assert_almost_equal(entropies[1], 0.0, decimal=5)
    # Parent 2, 3: no data => entropy = 0
    assert entropies[2] == 0.0
    assert entropies[3] == 0.0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_intra_parent_diversity_binary():
    """Binary L1 distribution gives 1 bit entropy."""
    counts = np.zeros((2, 4), dtype=np.int64)
    counts[0, 0] = 50
    counts[0, 1] = 50

    fig, entropies = compute_intra_parent_diversity(counts)
    np.testing.assert_almost_equal(entropies[0], 1.0, decimal=5)
    import matplotlib.pyplot as plt

    plt.close(fig)


# =============================================================================
# 3c. Hierarchical Transition Tests
# =============================================================================


def test_hierarchical_transitions_basic():
    """Transition rates match hand-computed values."""
    # L0: [0, 0, 0, 1, 1] => 1 L0 transition out of 4 steps
    # L1: [0, 1, 0, 0, 1] => 3 L1 transitions out of 4 steps
    # Within L0-same frames: t=1(L0 same), t=2(L0 same), t=4(L0 same)
    #   L1 transitions within L0: t=1(0→1), t=2(1→0), t=4(0→1) => 3/3
    result = _make_result([0, 0, 0, 1, 1], [0, 1, 0, 0, 1])
    fig, rates, within_trans = compute_hierarchical_transitions([result], num_codes=4)

    assert rates["l0_transition_rate"] == pytest.approx(1 / 4)
    assert rates["l1_transition_rate"] == pytest.approx(3 / 4)
    assert rates["l1_transition_rate_within_l0"] == pytest.approx(3 / 3)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_hierarchical_transitions_no_l0_change():
    """When L0 never changes, within-L0 rate equals unconditional rate."""
    result = _make_result([0, 0, 0, 0], [0, 1, 1, 0])
    fig, rates, _ = compute_hierarchical_transitions([result], num_codes=4)

    assert rates["l0_transition_rate"] == 0.0
    assert rates["l1_transition_rate"] == rates["l1_transition_rate_within_l0"]
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_hierarchical_transitions_within_parent_matrix():
    """Within-parent transition matrix accumulates correctly."""
    # L0: [0, 0, 0], L1: [0, 1, 2]
    # Within L0-same: t=1(L1: 0→1), t=2(L1: 1→2)
    result = _make_result([0, 0, 0], [0, 1, 2])
    _, _, within_trans = compute_hierarchical_transitions([result], num_codes=4)

    assert within_trans[0, 1] == 1  # L1: 0→1
    assert within_trans[1, 2] == 1  # L1: 1→2
    assert within_trans.sum() == 2
    import matplotlib.pyplot as plt

    plt.close("all")


# =============================================================================
# Pipeline Entry Point Tests
# =============================================================================


def test_run_rvq_analysis_skips_without_rvq(tmp_path):
    """Pipeline returns empty dict when no rvq_indices."""
    result = InferenceResult(
        clip_idx=0,
        code_indices=np.array([0, 1, 2]),
        qpos=np.zeros((3, 10)),
        qvel=np.zeros((3, 10)),
        rewards=np.zeros(3),
    )
    paths = run_rvq_analysis([result], num_codes=4, output_dir=tmp_path)
    assert paths == {}


def test_run_rvq_analysis_produces_figures(tmp_path):
    """Pipeline produces all 3 figures when rvq_indices are present."""
    result = _make_result(
        [0, 0, 1, 1, 2, 2, 0, 0] * 5,
        [0, 1, 0, 1, 0, 1, 0, 1] * 5,
    )
    paths = run_rvq_analysis([result], num_codes=4, output_dir=tmp_path)

    assert "parent_child_heatmap" in paths
    assert "intra_parent_diversity" in paths
    assert "hierarchical_transitions" in paths
    # Check files exist
    for p in paths.values():
        assert Path(p).exists()


def test_run_rvq_analysis_respects_config(tmp_path):
    """Pipeline respects config flags to disable individual analyses."""
    result = _make_result([0, 0, 1, 1], [0, 1, 0, 1])
    paths = run_rvq_analysis(
        [result],
        num_codes=4,
        output_dir=tmp_path,
        cfg={
            "parent_child_heatmap": True,
            "intra_parent_diversity": False,
            "hierarchical_transitions": False,
        },
    )
    assert "parent_child_heatmap" in paths
    assert "intra_parent_diversity" not in paths
    assert "hierarchical_transitions" not in paths
