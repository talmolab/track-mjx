"""Video rendering utilities with Nature paper style overlays for VQ-VAE analysis.

This module provides functions for rendering rollouts to video with
informative overlays showing codebook indices and transition patterns.
"""

import logging
from pathlib import Path
from typing import Any, Sequence

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def _build_stacked_bars(
    width: int,
    current_frame_idx: int,
    indices_per_depth: list[np.ndarray],
    code_colors: np.ndarray,
    bar_height: int = 30,
    separator_height: int = 2,
    playhead_width: int = 3,
) -> np.ndarray:
    """Build a stacked bar image for multi-depth RVQ timelines.

    Each depth level gets its own colored bar, stacked top-to-bottom
    (depth 0 on top, depth D-1 on bottom) with thin separators.

    Args:
        width: Width of the bar in pixels.
        current_frame_idx: Current frame index for playhead.
        indices_per_depth: List of D arrays, each shape [T].
        code_colors: Color for each code, shape [num_codes, 3].
        bar_height: Height of each individual bar in pixels.
        separator_height: Height of separator between bars.
        playhead_width: Width of the playhead marker.

    Returns:
        Stacked bar image, shape [total_height, width, 3].
    """
    n_depths = len(indices_per_depth)
    total_height = n_depths * bar_height + (n_depths - 1) * separator_height
    bar_img = np.ones((total_height, width, 3), dtype=np.uint8) * 255

    num_frames = len(indices_per_depth[0])
    playhead_x = int(current_frame_idx * width / num_frames)
    playhead_x = min(playhead_x, width - playhead_width)

    for d, depth_indices in enumerate(indices_per_depth):
        y_start = d * (bar_height + separator_height)

        # Draw code segments for this depth
        for j, code_idx in enumerate(depth_indices):
            x_start = int(j * width / num_frames)
            x_end = int((j + 1) * width / num_frames)
            color = code_colors[int(code_idx) % len(code_colors)]
            bar_img[y_start : y_start + bar_height, x_start:x_end] = color

        # Draw playhead
        if playhead_x > 0:
            bar_img[y_start : y_start + bar_height, playhead_x - 1 : playhead_x] = [
                50,
                50,
                50,
            ]
        bar_img[
            y_start : y_start + bar_height,
            playhead_x : playhead_x + playhead_width,
        ] = [255, 255, 255]
        if playhead_x + playhead_width < width:
            bar_img[
                y_start : y_start + bar_height,
                playhead_x + playhead_width : playhead_x + playhead_width + 1,
            ] = [50, 50, 50]

        # Draw separator below (except for last bar)
        if d < n_depths - 1:
            sep_y = y_start + bar_height
            bar_img[sep_y : sep_y + separator_height, :] = [50, 50, 50]

    return bar_img


def _add_multi_depth_code_label(
    frame: np.ndarray,
    current_frame_idx: int,
    indices_per_depth: list[np.ndarray],
    code_colors: np.ndarray,
    font_size: int = 16,
) -> np.ndarray:
    """Add a multi-depth code label badge (e.g. 'L0:5 L1:12') to frame.

    Args:
        frame: Input frame as numpy array, shape [H, W, 3].
        current_frame_idx: Current frame index.
        indices_per_depth: List of D index arrays.
        code_colors: Color for each code, shape [num_codes, 3].
        font_size: Font size for label.

    Returns:
        Frame with code label overlay.
    """
    parts = []
    for d, depth_indices in enumerate(indices_per_depth):
        if current_frame_idx < len(depth_indices):
            parts.append(f"L{d}:{int(depth_indices[current_frame_idx])}")
    if not parts:
        return frame

    label_text = " ".join(parts)
    # Use depth-0 code color for the badge background
    code_0 = int(indices_per_depth[0][current_frame_idx])
    color = code_colors[code_0 % len(code_colors)]

    label_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(label_img)
    font = _get_font(font_size, bold=True)

    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 6
    box_x, box_y = 10, 10

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

    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    draw.text((box_x, box_y), label_text, font=font, fill=text_color)

    return np.array(label_img)


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
    indices_per_depth: list[np.ndarray] | None = None,
) -> str:
    """Render rollout states to video with Nature-style overlays.

    Supports multi-depth RVQ: when ``indices_per_depth`` is provided
    (list of D arrays), stacked timeline bars are drawn (one per depth).
    Falls back to single bar when only ``indices`` is given.

    Args:
        env: Environment with render method.
        rollout_states: Sequence of environment states to render.
        output_path: Path to save output video.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        indices: Optional codebook indices per frame (depth-0 only).
        num_codes: Total number of codes.
        rewards: Optional rewards per frame.
        extra_info: Optional list of dicts with extra info per frame.
        code_bar_height: Height of the code color bar.
        clip_idx: Optional clip index to display.
        show_clip_idx: Whether to show clip index.
        indices_per_depth: Optional list of D index arrays for multi-depth
            RVQ. When provided, stacked bars are drawn and ``indices`` is
            ignored.

    Returns:
        Path to saved video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_states = len(rollout_states)
    logging.info(f"  Rendering {num_states} frames from environment...")

    frames = env.render(rollout_states, camera=camera, height=height, width=width)
    logging.info(f"  Rendered {len(frames)} frames")

    # Determine whether to use multi-depth or single-depth rendering
    use_multi_depth = indices_per_depth is not None and len(indices_per_depth) > 1

    code_colors = None
    has_indices = use_multi_depth or indices is not None
    if has_indices:
        if num_codes is None:
            if use_multi_depth:
                num_codes = int(max(np.max(a) for a in indices_per_depth)) + 1
            else:
                num_codes = int(np.max(indices)) + 1
        code_colors = get_nature_colormap(num_codes)

    logging.info("  Adding overlays...")
    processed_frames = []

    for i, frame in enumerate(frames):
        if use_multi_depth:
            # Build stacked bars and append below the frame
            bar_img = _build_stacked_bars(
                width=frame.shape[1],
                current_frame_idx=i,
                indices_per_depth=indices_per_depth,
                code_colors=code_colors,
                bar_height=code_bar_height,
            )
            frame = np.vstack([frame, bar_img])

            # Add multi-depth code label
            frame = _add_multi_depth_code_label(
                frame,
                current_frame_idx=i,
                indices_per_depth=indices_per_depth,
                code_colors=code_colors,
            )
        elif indices is not None:
            # Single-depth: original behavior
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
                frame, lines, start_position=(10, 50 if has_indices else 10)
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
