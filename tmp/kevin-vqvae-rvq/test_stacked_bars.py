"""Tests for multi-depth stacked bar rendering.

Verifies:
- _build_stacked_bars output shape (total_height, width, 3)
- Separator pixels between bars are dark
- Playhead drawn as white column in each bar region
- _add_multi_depth_code_label returns frame with same or larger dimensions
- Backward compat: indices_per_depth=None falls back to single-bar path
"""

import sys

sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx/vqvae_jax")
sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx")

import numpy as np
import pytest

from analysis.rendering import (
    _add_multi_depth_code_label,
    _build_stacked_bars,
    add_code_transition_bar,
    get_nature_colormap,
)


@pytest.fixture
def code_colors():
    """Nature colormap with 8 codes."""
    return get_nature_colormap(8)


@pytest.fixture
def two_depth_indices():
    """Two depth levels, 50 timesteps each, codes in range [0, 8)."""
    rng = np.random.RandomState(42)
    depth_0 = rng.randint(0, 8, size=50).astype(np.int32)
    depth_1 = rng.randint(0, 8, size=50).astype(np.int32)
    return [depth_0, depth_1]


@pytest.fixture
def three_depth_indices():
    """Three depth levels, 50 timesteps each."""
    rng = np.random.RandomState(99)
    return [rng.randint(0, 8, size=50).astype(np.int32) for _ in range(3)]


@pytest.fixture
def synthetic_frame():
    """100x200 black frame."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


# =========================================================================
# _build_stacked_bars shape tests
# =========================================================================


def test_stacked_bars_shape_2_depths(code_colors, two_depth_indices):
    """Output height = 2*bar_height + 1*separator_height, width matches input."""
    bar_height = 30
    separator_height = 2
    width = 200

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=0,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=separator_height,
    )

    expected_height = 2 * bar_height + 1 * separator_height  # 62
    assert bar_img.shape == (
        expected_height,
        width,
        3,
    ), f"Expected shape ({expected_height}, {width}, 3), got {bar_img.shape}"


def test_stacked_bars_shape_3_depths(code_colors, three_depth_indices):
    """Output height = 3*bar_height + 2*separator_height."""
    bar_height = 20
    separator_height = 4
    width = 150

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=10,
        indices_per_depth=three_depth_indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=separator_height,
    )

    expected_height = 3 * bar_height + 2 * separator_height  # 68
    assert bar_img.shape == (
        expected_height,
        width,
        3,
    ), f"Expected shape ({expected_height}, {width}, 3), got {bar_img.shape}"


def test_stacked_bars_shape_1_depth(code_colors):
    """Single depth: no separator, height = bar_height."""
    bar_height = 25
    width = 180
    indices = [np.array([0, 1, 2, 3, 4], dtype=np.int32)]

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=2,
        indices_per_depth=indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=2,
    )

    expected_height = 1 * bar_height + 0 * 2  # 25
    assert bar_img.shape == (expected_height, width, 3)


def test_stacked_bars_dtype(code_colors, two_depth_indices):
    """Output should be uint8."""
    bar_img = _build_stacked_bars(
        width=200,
        current_frame_idx=0,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )
    assert bar_img.dtype == np.uint8


# =========================================================================
# Separator tests
# =========================================================================


def test_stacked_bars_separator_is_dark(code_colors, two_depth_indices):
    """With 2 depths, separator pixels between bars should be dark (<=50 per channel)."""
    bar_height = 30
    separator_height = 2
    width = 200

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=25,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=separator_height,
    )

    # Separator region: rows [bar_height, bar_height + separator_height)
    sep_region = bar_img[bar_height : bar_height + separator_height, :, :]

    # All separator pixels should be dark ([50, 50, 50] per implementation)
    assert np.all(sep_region <= 50), (
        f"Separator pixels should be dark (<=50), "
        f"max value found: {sep_region.max()}"
    )


def test_stacked_bars_no_separator_single_depth(code_colors):
    """Single depth should have no separator at all."""
    bar_height = 30
    indices = [np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)]

    bar_img = _build_stacked_bars(
        width=200,
        current_frame_idx=0,
        indices_per_depth=indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=2,
    )

    # Image height is exactly bar_height, no separator
    assert bar_img.shape[0] == bar_height


# =========================================================================
# Playhead tests
# =========================================================================


def test_stacked_bars_playhead_is_white(code_colors, two_depth_indices):
    """Playhead should be drawn as a white column in each bar region."""
    bar_height = 30
    separator_height = 2
    width = 200
    playhead_width = 3
    current_frame_idx = 25
    num_frames = len(two_depth_indices[0])

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=current_frame_idx,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=separator_height,
        playhead_width=playhead_width,
    )

    # Compute expected playhead x position (same logic as the implementation)
    playhead_x = int(current_frame_idx * width / num_frames)
    playhead_x = min(playhead_x, width - playhead_width)

    # Check playhead column in depth 0 bar
    depth_0_playhead = bar_img[
        0:bar_height, playhead_x : playhead_x + playhead_width, :
    ]
    assert np.all(depth_0_playhead == 255), (
        f"Playhead in depth 0 should be white (255), "
        f"got values: {np.unique(depth_0_playhead)}"
    )

    # Check playhead column in depth 1 bar
    y1_start = bar_height + separator_height
    depth_1_playhead = bar_img[
        y1_start : y1_start + bar_height, playhead_x : playhead_x + playhead_width, :
    ]
    assert np.all(depth_1_playhead == 255), (
        f"Playhead in depth 1 should be white (255), "
        f"got values: {np.unique(depth_1_playhead)}"
    )


def test_stacked_bars_playhead_dark_border(code_colors, two_depth_indices):
    """Playhead should have dark border pixels on left and right."""
    bar_height = 30
    width = 200
    playhead_width = 3
    current_frame_idx = 25
    num_frames = len(two_depth_indices[0])

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=current_frame_idx,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
        bar_height=bar_height,
        playhead_width=playhead_width,
    )

    playhead_x = int(current_frame_idx * width / num_frames)
    playhead_x = min(playhead_x, width - playhead_width)

    # Left border of playhead (if not at edge)
    if playhead_x > 0:
        left_border = bar_img[0:bar_height, playhead_x - 1 : playhead_x, :]
        assert np.all(
            left_border == 50
        ), f"Left border should be [50,50,50], got {np.unique(left_border)}"

    # Right border of playhead (if not at edge)
    if playhead_x + playhead_width < width:
        right_border = bar_img[
            0:bar_height,
            playhead_x + playhead_width : playhead_x + playhead_width + 1,
            :,
        ]
        assert np.all(
            right_border == 50
        ), f"Right border should be [50,50,50], got {np.unique(right_border)}"


# =========================================================================
# _add_multi_depth_code_label tests
# =========================================================================


def test_multi_depth_code_label_same_dimensions(
    synthetic_frame, code_colors, two_depth_indices
):
    """Label overlay should preserve frame dimensions (H, W, 3)."""
    result = _add_multi_depth_code_label(
        frame=synthetic_frame,
        current_frame_idx=10,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )

    assert (
        result.shape == synthetic_frame.shape
    ), f"Expected shape {synthetic_frame.shape}, got {result.shape}"


def test_multi_depth_code_label_modifies_pixels(
    synthetic_frame, code_colors, two_depth_indices
):
    """Label overlay should actually change some pixels (badge is drawn)."""
    result = _add_multi_depth_code_label(
        frame=synthetic_frame.copy(),
        current_frame_idx=10,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )

    # On a black frame, adding a colored badge should produce non-zero pixels
    assert not np.array_equal(
        result, synthetic_frame
    ), "Label overlay should modify the frame"


def test_multi_depth_code_label_out_of_bounds(
    synthetic_frame, code_colors, two_depth_indices
):
    """When current_frame_idx >= len(indices), return unchanged frame."""
    result = _add_multi_depth_code_label(
        frame=synthetic_frame.copy(),
        current_frame_idx=999,  # Way past end
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )

    assert np.array_equal(
        result, synthetic_frame
    ), "Out-of-bounds index should return unchanged frame"


def test_multi_depth_code_label_dtype(synthetic_frame, code_colors, two_depth_indices):
    """Output should be uint8."""
    result = _add_multi_depth_code_label(
        frame=synthetic_frame,
        current_frame_idx=0,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )
    assert result.dtype == np.uint8


# =========================================================================
# Backward compatibility: single-depth path
# =========================================================================


def test_single_bar_path_shape(synthetic_frame, code_colors):
    """add_code_transition_bar overlays on the existing frame (no height increase)."""
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    bar_height = 40

    result = add_code_transition_bar(
        frame=synthetic_frame.copy(),
        current_frame_idx=3,
        all_indices=indices,
        code_colors=code_colors,
        bar_height=bar_height,
    )

    # Single bar path overlays on the frame, so shape stays the same
    assert (
        result.shape[0] == synthetic_frame.shape[0]
    ), "Single-bar path should not change frame height"
    assert (
        result.shape[1] == synthetic_frame.shape[1]
    ), "Single-bar path should not change frame width"


def test_multi_depth_path_adds_height(synthetic_frame, code_colors, two_depth_indices):
    """Multi-depth path vstacks bars below frame, increasing total height."""
    bar_height = 30
    separator_height = 2

    # Build stacked bars like render_rollout_to_video does
    bar_img = _build_stacked_bars(
        width=synthetic_frame.shape[1],
        current_frame_idx=5,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=separator_height,
    )
    combined = np.vstack([synthetic_frame, bar_img])

    expected_bar_height = 2 * bar_height + 1 * separator_height
    expected_total_height = synthetic_frame.shape[0] + expected_bar_height

    assert (
        combined.shape[0] == expected_total_height
    ), f"Expected total height {expected_total_height}, got {combined.shape[0]}"
    assert combined.shape[1] == synthetic_frame.shape[1], "Width should be unchanged"


# =========================================================================
# Edge cases
# =========================================================================


def test_stacked_bars_first_frame(code_colors, two_depth_indices):
    """Playhead at frame 0 should not go negative."""
    bar_img = _build_stacked_bars(
        width=200,
        current_frame_idx=0,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )
    # No assertion on specific pixels, just that it runs without error
    assert bar_img.shape[2] == 3


def test_stacked_bars_last_frame(code_colors, two_depth_indices):
    """Playhead at last frame should not exceed width."""
    last_idx = len(two_depth_indices[0]) - 1
    bar_img = _build_stacked_bars(
        width=200,
        current_frame_idx=last_idx,
        indices_per_depth=two_depth_indices,
        code_colors=code_colors,
    )
    assert bar_img.shape[2] == 3


def test_stacked_bars_color_correctness(code_colors):
    """Verify that bar pixels match the code_colors for known indices."""
    width = 100
    # Create indices with a single constant code per depth
    depth_0 = np.full(10, 3, dtype=np.int32)  # All code 3
    depth_1 = np.full(10, 5, dtype=np.int32)  # All code 5
    indices_per_depth = [depth_0, depth_1]
    bar_height = 20
    separator_height = 2
    playhead_width = 3

    bar_img = _build_stacked_bars(
        width=width,
        current_frame_idx=0,
        indices_per_depth=indices_per_depth,
        code_colors=code_colors,
        bar_height=bar_height,
        separator_height=separator_height,
        playhead_width=playhead_width,
    )

    # Sample a pixel in depth 0 bar far from playhead
    # Playhead is at x=0, so sample at x=width//2
    sample_x = width // 2
    sample_y = bar_height // 2  # Middle of depth 0 bar
    pixel_d0 = bar_img[sample_y, sample_x, :]
    expected_d0 = code_colors[3]
    assert np.array_equal(
        pixel_d0, expected_d0
    ), f"Depth 0 pixel should be color of code 3: {expected_d0}, got {pixel_d0}"

    # Sample a pixel in depth 1 bar far from playhead
    sample_y_d1 = bar_height + separator_height + bar_height // 2
    pixel_d1 = bar_img[sample_y_d1, sample_x, :]
    expected_d1 = code_colors[5]
    assert np.array_equal(
        pixel_d1, expected_d1
    ), f"Depth 1 pixel should be color of code 5: {expected_d1}, got {pixel_d1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
