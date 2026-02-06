"""Video rendering utilities with Nature paper style overlays for VQ-VAE analysis.

This module provides functions for rendering rollouts to video with
informative overlays showing codebook indices and transition patterns.
Supports both individual clips and grid montages, as well as community-based
visualization for large codebooks.
"""

import logging
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .community_analysis import CommunityStructure
    from .inference_cache import InferenceResult


# =============================================================================
# FONT LOADING
# =============================================================================


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font, falling back gracefully.

    Args:
        size: Font size in pixels.
        bold: Whether to use bold variant.

    Returns:
        Loaded font.
    """
    font_paths = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue

    return ImageFont.load_default()


# =============================================================================
# COLORMAP GENERATION
# =============================================================================


def get_code_colormap(num_codes: int) -> np.ndarray:
    """Generate a perceptually uniform colormap for codebook indices.

    Uses a combination of matplotlib's qualitative colormaps for good
    discriminability between adjacent codes.

    Args:
        num_codes: Number of codes in the codebook.

    Returns:
        Array of RGB colors, shape [num_codes, 3], values 0-255.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if num_codes <= 20:
        cmap = plt.cm.tab20
        colors = [cmap(i / 20) for i in range(num_codes)]
    elif num_codes <= 60:
        # Combine tab20, tab20b, tab20c for more colors
        cmap1, cmap2, cmap3 = plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c
        colors = []
        for i in range(num_codes):
            if i < 20:
                colors.append(cmap1(i / 20))
            elif i < 40:
                colors.append(cmap2((i - 20) / 20))
            else:
                colors.append(cmap3((i - 40) / 20))
    else:
        # Use HSV for large codebooks
        colors = [plt.cm.hsv(i / num_codes) for i in range(num_codes)]

    return np.array(
        [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)] for c in colors]
    )


def get_nature_colormap(num_codes: int) -> np.ndarray:
    """Generate a Nature-style colormap with muted, professional colors.

    Args:
        num_codes: Number of codes in the codebook.

    Returns:
        Array of RGB colors, shape [num_codes, 3], values 0-255.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use a perceptually uniform colormap with better aesthetics
    if num_codes <= 10:
        cmap = plt.cm.Set3
    elif num_codes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    colors = [cmap(i / max(num_codes - 1, 1)) for i in range(num_codes)]
    return np.array(
        [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)] for c in colors]
    )


# =============================================================================
# TEXT OVERLAYS
# =============================================================================


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 10),
    font_size: int = 20,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int, int] = (0, 0, 0, 180),
    padding: int = 5,
) -> np.ndarray:
    """Add text overlay with semi-transparent background.

    Args:
        frame: Input frame as numpy array, shape [H, W, 3].
        text: Text to overlay.
        position: (x, y) position for text.
        font_size: Font size in pixels.
        text_color: RGB color for text.
        bg_color: RGBA color for background.
        padding: Padding around text.

    Returns:
        Frame with text overlay.
    """
    img = Image.fromarray(frame.astype(np.uint8))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_size)

    bbox = draw.textbbox(position, text, font=font)
    bg_rect = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )
    draw.rectangle(bg_rect, fill=bg_color)
    draw.text(position, text, font=font, fill=text_color + (255,))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def add_multi_line_overlay(
    frame: np.ndarray,
    lines: list[str],
    start_position: tuple[int, int] = (10, 10),
    font_size: int = 18,
    line_spacing: int = 5,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int, int] = (0, 0, 0, 180),
    padding: int = 5,
) -> np.ndarray:
    """Add multi-line text overlay.

    Args:
        frame: Input frame as numpy array.
        lines: List of text lines to overlay.
        start_position: (x, y) position for first line.
        font_size: Font size in pixels.
        line_spacing: Spacing between lines.
        text_color: RGB color for text.
        bg_color: RGBA color for background.
        padding: Padding around text block.

    Returns:
        Frame with text overlay.
    """
    img = Image.fromarray(frame.astype(np.uint8))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_size)

    x, y = start_position
    max_width = 0
    total_height = 0
    line_bboxes = []

    for line in lines:
        bbox = draw.textbbox((x, y + total_height), line, font=font)
        line_bboxes.append((x, y + total_height, bbox))
        max_width = max(max_width, bbox[2] - bbox[0])
        total_height += bbox[3] - bbox[1] + line_spacing

    bg_rect = (
        x - padding,
        y - padding,
        x + max_width + padding,
        y + total_height + padding,
    )
    draw.rectangle(bg_rect, fill=bg_color)

    for line, (lx, ly, _) in zip(lines, line_bboxes):
        draw.text((lx, ly), line, font=font, fill=text_color + (255,))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


# =============================================================================
# CODE TRANSITION BAR (NATURE PAPER STYLE)
# =============================================================================


def add_code_transition_bar(
    frame: np.ndarray,
    current_frame_idx: int,
    all_indices: np.ndarray,
    code_colors: np.ndarray,
    bar_height: int = 40,
    playhead_width: int = 3,
    show_playhead: bool = True,
    show_code_label: bool = True,
    label_position: str = "top_left",
    font_size: int = 16,
    border_width: int = 1,
) -> np.ndarray:
    """Add a Nature-style code transition timeline bar.

    Creates a clean, professional-looking timeline showing the sequence
    of codes as colored segments with a playhead marker.

    Args:
        frame: Input frame as numpy array, shape [H, W, 3].
        current_frame_idx: Current frame index in the sequence.
        all_indices: Array of all code indices for the full sequence.
        code_colors: Color for each code, shape [num_codes, 3].
        bar_height: Height of the timeline bar in pixels.
        playhead_width: Width of the playhead marker.
        show_playhead: Whether to show the playhead marker.
        show_code_label: Whether to show the current code number.
        label_position: Position for code label.
        font_size: Font size for code label.
        border_width: Width of border around bar.

    Returns:
        Frame with timeline overlay.
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    num_frames = len(all_indices)
    y_start = h - bar_height

    # Create timeline bar with white border
    timeline = np.ones((bar_height, w, 3), dtype=np.uint8) * 255

    # Draw border at top
    timeline[:border_width, :] = [200, 200, 200]

    # Draw code segments
    bar_content_start = border_width
    bar_content_height = bar_height - border_width

    for i, code_idx in enumerate(all_indices):
        x_start = int(i * w / num_frames)
        x_end = int((i + 1) * w / num_frames)
        color = code_colors[code_idx % len(code_colors)]
        timeline[bar_content_start:, x_start:x_end] = color

    # Overlay timeline on frame
    result[y_start : y_start + bar_height, :] = timeline

    # Draw playhead
    if show_playhead:
        playhead_x = int(current_frame_idx * w / num_frames)
        playhead_x = min(playhead_x, w - playhead_width)

        # White playhead with dark border
        if playhead_x > 0:
            result[y_start:, playhead_x - 1 : playhead_x] = [50, 50, 50]
        result[y_start:, playhead_x : playhead_x + playhead_width] = [255, 255, 255]
        if playhead_x + playhead_width < w:
            result[
                y_start:, playhead_x + playhead_width : playhead_x + playhead_width + 1
            ] = [50, 50, 50]

    # Add code label
    if show_code_label and current_frame_idx < len(all_indices):
        current_code = int(all_indices[current_frame_idx])
        color = code_colors[current_code % len(code_colors)]

        label_img = Image.fromarray(result)
        draw = ImageDraw.Draw(label_img)
        font = _get_font(font_size, bold=True)

        label_text = f"Code {current_code}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position based on config
        padding = 6
        if label_position == "top_left":
            box_x, box_y = 10, 10
        elif label_position == "top_right":
            box_x, box_y = w - text_width - padding * 2 - 10, 10
        elif label_position == "bottom_left":
            box_x, box_y = 10, y_start - text_height - padding * 2 - 10
        else:  # bottom_right
            box_x, box_y = (
                w - text_width - padding * 2 - 10,
                y_start - text_height - padding * 2 - 10,
            )

        # Draw rounded rectangle background
        draw.rounded_rectangle(
            [
                box_x - padding,
                box_y - padding,
                box_x + text_width + padding,
                box_y + text_height + padding,
            ],
            radius=4,
            fill=tuple(color),
        )

        # Choose text color for contrast
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        draw.text((box_x, box_y), label_text, font=font, fill=text_color)

        result = np.array(label_img)

    return result


# =============================================================================
# VIDEO RENDERING
# =============================================================================


def render_rollout_to_video(
    env: Any,
    rollout_states: Sequence[Any],
    output_path: str | Path,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    indices: np.ndarray | None = None,
    num_codes: int | None = None,
    rewards: np.ndarray | None = None,
    extra_info: list[dict[str, Any]] | None = None,
    code_bar_height: int = 40,
    clip_idx: int | None = None,
    show_clip_idx: bool = True,
) -> str:
    """Render rollout states to video with Nature-style overlays.

    Args:
        env: Environment with render method.
        rollout_states: Sequence of environment states to render.
        output_path: Path to save output video.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        indices: Optional codebook indices per frame.
        num_codes: Total number of codes.
        rewards: Optional rewards per frame.
        extra_info: Optional list of dicts with extra info per frame.
        code_bar_height: Height of the code color bar.
        clip_idx: Optional clip index to display.
        show_clip_idx: Whether to show clip index.

    Returns:
        Path to saved video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_states = len(rollout_states)
    logging.info(f"  Rendering {num_states} frames from environment...")

    frames = env.render(rollout_states, camera=camera, height=height, width=width)
    logging.info(f"  Rendered {len(frames)} frames")

    code_colors = None
    if indices is not None:
        if num_codes is None:
            num_codes = int(np.max(indices)) + 1
        code_colors = get_nature_colormap(num_codes)

    logging.info("  Adding overlays...")
    processed_frames = []

    for i, frame in enumerate(frames):
        # Add code transition bar
        if indices is not None:
            frame = add_code_transition_bar(
                frame,
                current_frame_idx=i,
                all_indices=indices,
                code_colors=code_colors,
                bar_height=code_bar_height,
            )

        # Add text overlays
        lines = []
        if show_clip_idx and clip_idx is not None:
            lines.append(f"Clip {clip_idx}")

        if rewards is not None and i < len(rewards):
            lines.append(f"Reward: {float(rewards[i]):.3f}")

        if extra_info is not None and i < len(extra_info):
            for key, value in extra_info[i].items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.3f}")
                else:
                    lines.append(f"{key}: {value}")

        if lines:
            # Position text below the code label area
            frame = add_multi_line_overlay(
                frame, lines, start_position=(10, 50 if indices is not None else 10)
            )

        processed_frames.append(frame)

    logging.info(f"  Writing video ({len(processed_frames)} frames at {fps} fps)...")
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in processed_frames:
            writer.append_data(frame)

    logging.info(f"  Video saved to {output_path}")
    return str(output_path)


def render_per_code_videos(
    env: Any,
    rollout_states: Sequence[Any],
    indices: np.ndarray,
    output_dir: str | Path,
    num_codes: int,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    min_frames_per_code: int = 5,
    code_bar_height: int = 40,
) -> dict[int, str]:
    """Render separate videos for each code, showing only frames where that code was active.

    This helps visualize what behavior each discrete code corresponds to by showing
    all the frames where that particular code was selected by the policy.

    Args:
        env: Environment with render method.
        rollout_states: Sequence of environment states from the rollout.
        indices: Array of code indices for each frame, shape [num_frames].
        output_dir: Directory to save per-code videos.
        num_codes: Total number of codes in the codebook.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        min_frames_per_code: Minimum frames required to create a video for a code.
        code_bar_height: Height of the code indicator bar.

    Returns:
        Dict mapping code index to video path for codes that had enough frames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render all frames first
    logging.info(f"  Rendering {len(rollout_states)} frames for per-code videos...")
    all_frames = env.render(rollout_states, camera=camera, height=height, width=width)

    # Get colormap
    code_colors = get_nature_colormap(num_codes)

    # Group frame indices by code
    code_to_frame_indices: dict[int, list[int]] = {i: [] for i in range(num_codes)}
    for frame_idx, code_idx in enumerate(indices):
        code_to_frame_indices[int(code_idx)].append(frame_idx)

    # Create video for each code that has enough frames
    output_paths: dict[int, str] = {}
    codes_used = [
        code
        for code, frames in code_to_frame_indices.items()
        if len(frames) >= min_frames_per_code
    ]

    logging.info(
        f"  Creating videos for {len(codes_used)} codes with >= {min_frames_per_code} frames"
    )

    for code_idx in codes_used:
        frame_indices = code_to_frame_indices[code_idx]
        code_color = code_colors[code_idx]

        # Collect frames for this code
        code_frames = []
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx < len(all_frames):
                frame = all_frames[frame_idx].copy()

                # Add code label overlay
                frame = add_text_overlay(
                    frame,
                    f"Code {code_idx} | Frame {frame_idx}",
                    position=(10, 10),
                    font_size=16,
                    bg_color=(
                        int(code_color[0]),
                        int(code_color[1]),
                        int(code_color[2]),
                        200,
                    ),
                    text_color=(255, 255, 255) if sum(code_color) < 384 else (0, 0, 0),
                )

                # Add progress indicator bar at bottom
                bar = np.zeros((code_bar_height, width, 3), dtype=np.uint8)
                bar[:] = code_color
                # Add progress marker
                progress = int((i / max(len(frame_indices) - 1, 1)) * width)
                bar[:, max(0, progress - 2) : min(width, progress + 2)] = [
                    255,
                    255,
                    255,
                ]

                # Combine frame and bar
                combined = np.vstack([frame, bar])
                code_frames.append(combined)

        if code_frames:
            video_path = output_dir / f"code_{code_idx}.mp4"
            with imageio.get_writer(str(video_path), fps=fps) as writer:
                for frame in code_frames:
                    writer.append_data(frame)
            output_paths[code_idx] = str(video_path)
            logging.info(
                f"    Code {code_idx}: {len(code_frames)} frames -> {video_path.name}"
            )

    return output_paths


# =============================================================================
# COMMUNITY-BASED VISUALIZATION
# =============================================================================

# Community color palette (8 distinct, saturated colors)
COMMUNITY_COLORS = np.array(
    [
        [66, 133, 244],  # Blue
        [234, 67, 53],  # Red
        [251, 188, 5],  # Yellow
        [52, 168, 83],  # Green
        [155, 89, 182],  # Purple
        [26, 188, 156],  # Teal
        [241, 196, 15],  # Gold
        [230, 126, 34],  # Orange
    ],
    dtype=np.uint8,
)


def get_community_colormap(n_communities: int) -> np.ndarray:
    """Get a colormap for community visualization.

    Args:
        n_communities: Number of communities.

    Returns:
        Array of RGB colors, shape [n_communities, 3], values 0-255.
    """
    if n_communities <= len(COMMUNITY_COLORS):
        return COMMUNITY_COLORS[:n_communities]

    # Generate additional colors via HSV
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = list(COMMUNITY_COLORS)
    for i in range(len(COMMUNITY_COLORS), n_communities):
        hue = (i - len(COMMUNITY_COLORS)) / (n_communities - len(COMMUNITY_COLORS) + 1)
        c = plt.cm.hsv(hue)
        colors.append([int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)])

    return np.array(colors, dtype=np.uint8)


def find_code_segments(
    indices: np.ndarray,
    min_segment_length: int = 20,
) -> dict[int, list[tuple[int, int]]]:
    """Find contiguous segments for each code that meet minimum length.

    Args:
        indices: Array of code indices, shape [T].
        min_segment_length: Minimum length for a segment to be included.

    Returns:
        Dict mapping code_idx -> list of (start, end) tuples for valid segments.
    """
    code_segments: dict[int, list[tuple[int, int]]] = {}

    if len(indices) == 0:
        return code_segments

    # Find all contiguous segments
    current_code = int(indices[0])
    segment_start = 0

    for i in range(1, len(indices)):
        if int(indices[i]) != current_code:
            # End of segment
            segment_length = i - segment_start
            if segment_length >= min_segment_length:
                if current_code not in code_segments:
                    code_segments[current_code] = []
                code_segments[current_code].append((segment_start, i))

            # Start new segment
            current_code = int(indices[i])
            segment_start = i

    # Handle last segment
    segment_length = len(indices) - segment_start
    if segment_length >= min_segment_length:
        if current_code not in code_segments:
            code_segments[current_code] = []
        code_segments[current_code].append((segment_start, len(indices)))

    return code_segments


def render_community_gallery(
    env: Any,
    all_rollout_states: list[Sequence[Any]],
    all_rollout_indices: list[np.ndarray],
    community_codes: list[int],
    community_id: int,
    output_path: str | Path,
    camera: str | None = None,
    cell_width: int = 320,
    cell_height: int = 240,
    fps: int = 50,
    min_segment_length: int = 20,
    max_codes_per_row: int = 4,
    max_frames_per_code: int = 100,
    num_codes: int | None = None,
) -> str | None:
    """Render a gallery video showing frames for each code in a community.

    Creates a grid where each cell shows frames from one code in the community.
    Only includes segments that meet the minimum length requirement.

    Args:
        env: Environment with render method.
        all_rollout_states: List of rollout state sequences.
        all_rollout_indices: List of code index arrays, each [T].
        community_codes: List of code indices in this community.
        community_id: ID of this community (for labeling).
        output_path: Path to save output video.
        camera: Camera name for rendering.
        cell_width: Width of each cell in the grid.
        cell_height: Height of each cell in the grid.
        fps: Frames per second.
        min_segment_length: Minimum segment length to include (default 20).
        max_codes_per_row: Maximum codes per row in the grid.
        max_frames_per_code: Maximum frames to show per code.
        num_codes: Total number of codes (for colormap).

    Returns:
        Path to saved video, or None if no valid segments found.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect valid segments for each code across all rollouts
    code_to_segments: dict[int, list[tuple[int, int, int]]] = {
        code: [] for code in community_codes
    }  # code -> [(rollout_idx, start, end), ...]

    for rollout_idx, indices in enumerate(all_rollout_indices):
        segments = find_code_segments(indices, min_segment_length)
        for code in community_codes:
            if code in segments:
                for start, end in segments[code]:
                    code_to_segments[code].append((rollout_idx, start, end))

    # Filter to codes that have at least one valid segment
    codes_with_segments = [
        code for code in community_codes if len(code_to_segments[code]) > 0
    ]

    if not codes_with_segments:
        logging.warning(
            f"Community {community_id}: No codes with segments >= {min_segment_length}"
        )
        return None

    logging.info(
        f"Community {community_id}: {len(codes_with_segments)}/{len(community_codes)} "
        f"codes have segments >= {min_segment_length} frames"
    )

    # Setup grid dimensions
    n_codes = len(codes_with_segments)
    grid_cols = min(n_codes, max_codes_per_row)
    grid_rows = (n_codes + grid_cols - 1) // grid_cols

    # Get colormap
    if num_codes is None:
        num_codes = max(community_codes) + 1
    code_colors = get_nature_colormap(num_codes)

    # Render frames for each code
    code_frames_list: list[list[np.ndarray]] = []

    for code in codes_with_segments:
        code_color = code_colors[code]
        segments = code_to_segments[code]

        # Collect frames from segments (up to max_frames_per_code)
        frames_to_render: list[tuple[int, int]] = []  # (rollout_idx, frame_idx)
        for rollout_idx, start, end in segments:
            for frame_idx in range(start, end):
                frames_to_render.append((rollout_idx, frame_idx))
                if len(frames_to_render) >= max_frames_per_code:
                    break
            if len(frames_to_render) >= max_frames_per_code:
                break

        # Group by rollout for efficient rendering
        rollout_frame_map: dict[int, list[int]] = {}
        for rollout_idx, frame_idx in frames_to_render:
            if rollout_idx not in rollout_frame_map:
                rollout_frame_map[rollout_idx] = []
            rollout_frame_map[rollout_idx].append(frame_idx)

        # Render frames
        code_frames: list[np.ndarray] = []
        for rollout_idx, frame_indices in rollout_frame_map.items():
            states = all_rollout_states[rollout_idx]
            # Render only the needed states
            states_to_render = [states[i] for i in frame_indices if i < len(states)]
            if not states_to_render:
                continue

            rendered = env.render(
                states_to_render,
                camera=camera,
                height=cell_height - 30,
                width=cell_width,
            )

            for i, frame in enumerate(rendered):
                frame_idx = frame_indices[i]
                # Add code label
                frame = add_text_overlay(
                    frame,
                    f"Code {code}",
                    position=(5, 5),
                    font_size=14,
                    bg_color=(
                        int(code_color[0]),
                        int(code_color[1]),
                        int(code_color[2]),
                        220,
                    ),
                    text_color=(255, 255, 255) if sum(code_color) < 384 else (0, 0, 0),
                    padding=4,
                )

                # Add colored bar at bottom
                bar = np.zeros((30, cell_width, 3), dtype=np.uint8)
                bar[:] = code_color
                combined = np.vstack([frame, bar])
                code_frames.append(combined)

        code_frames_list.append(code_frames)

    if not any(code_frames_list):
        logging.warning(f"Community {community_id}: No frames rendered")
        return None

    # Find max frames across codes
    max_frames = max(len(frames) for frames in code_frames_list)

    # Calculate grid dimensions
    padding = 4
    grid_width = grid_cols * cell_width + (grid_cols - 1) * padding
    grid_height = grid_rows * cell_height + (grid_rows - 1) * padding

    # Assemble grid frames
    grid_frames = []
    for frame_idx in range(max_frames):
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40  # Dark bg

        for code_num, code_frames in enumerate(code_frames_list):
            if not code_frames:
                continue

            row = code_num // grid_cols
            col = code_num % grid_cols

            # Use last frame if past end, or blank if no frames
            if frame_idx < len(code_frames):
                frame = code_frames[frame_idx]
            else:
                frame = (
                    code_frames[-1]
                    if code_frames
                    else np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                )

            y_start = row * (cell_height + padding)
            x_start = col * (cell_width + padding)

            grid[y_start : y_start + cell_height, x_start : x_start + cell_width] = (
                frame
            )

        grid_frames.append(grid)

    # Add community label to all frames
    community_color = COMMUNITY_COLORS[community_id % len(COMMUNITY_COLORS)]
    for i, grid in enumerate(grid_frames):
        grid_frames[i] = add_text_overlay(
            grid,
            f"Community {community_id} ({len(codes_with_segments)} codes)",
            position=(10, grid_height - 35),
            font_size=16,
            bg_color=(
                int(community_color[0]),
                int(community_color[1]),
                int(community_color[2]),
                220,
            ),
            text_color=(255, 255, 255),
            padding=5,
        )

    # Save video
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in grid_frames:
            writer.append_data(frame)

    logging.info(
        f"Community {community_id}: Saved gallery ({n_codes} codes, "
        f"{max_frames} frames) to {output_path}"
    )
    return str(output_path)
