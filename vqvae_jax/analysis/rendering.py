"""Video rendering utilities with Nature paper style overlays for VQ-VAE analysis.

This module provides functions for rendering rollouts to video with
informative overlays showing codebook indices and transition patterns.
Supports both individual clips and grid montages.
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
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
    matplotlib.use('Agg')
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

    return np.array([[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)]
                     for c in colors])


def get_nature_colormap(num_codes: int) -> np.ndarray:
    """Generate a Nature-style colormap with muted, professional colors.

    Args:
        num_codes: Number of codes in the codebook.

    Returns:
        Array of RGB colors, shape [num_codes, 3], values 0-255.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Use a perceptually uniform colormap with better aesthetics
    if num_codes <= 10:
        cmap = plt.cm.Set3
    elif num_codes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    colors = [cmap(i / max(num_codes - 1, 1)) for i in range(num_codes)]
    return np.array([[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)]
                     for c in colors])


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
    bg_rect = (bbox[0] - padding, bbox[1] - padding,
               bbox[2] + padding, bbox[3] + padding)
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

    bg_rect = (x - padding, y - padding,
               x + max_width + padding, y + total_height + padding)
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
    result[y_start:y_start + bar_height, :] = timeline

    # Draw playhead
    if show_playhead:
        playhead_x = int(current_frame_idx * w / num_frames)
        playhead_x = min(playhead_x, w - playhead_width)

        # White playhead with dark border
        if playhead_x > 0:
            result[y_start:, playhead_x - 1:playhead_x] = [50, 50, 50]
        result[y_start:, playhead_x:playhead_x + playhead_width] = [255, 255, 255]
        if playhead_x + playhead_width < w:
            result[y_start:, playhead_x + playhead_width:playhead_x + playhead_width + 1] = [50, 50, 50]

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
            box_x, box_y = w - text_width - padding * 2 - 10, y_start - text_height - padding * 2 - 10

        # Draw rounded rectangle background
        draw.rounded_rectangle(
            [box_x - padding, box_y - padding,
             box_x + text_width + padding, box_y + text_height + padding],
            radius=4,
            fill=tuple(color)
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
                frame, lines,
                start_position=(10, 50 if indices is not None else 10)
            )

        processed_frames.append(frame)

    logging.info(f"  Writing video ({len(processed_frames)} frames at {fps} fps)...")
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in processed_frames:
            writer.append_data(frame)

    logging.info(f"  Video saved to {output_path}")
    return str(output_path)


# =============================================================================
# GRID RENDERING (NATURE PAPER STYLE)
# =============================================================================


def render_clips_grid(
    env: Any,
    clip_data: list[dict[str, Any]],
    output_path: str | Path,
    max_rows: int = 5,
    max_cols: int = 5,
    camera: str | None = None,
    cell_width: int = 320,
    cell_height: int = 240,
    fps: int = 50,
    num_codes: int | None = None,
    code_bar_height: int = 30,
    padding: int = 4,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[str]:
    """Render multiple clips in a grid layout (Nature paper style).

    Creates professional grid montages with a maximum of max_rows x max_cols
    clips per video. If more clips are provided, multiple videos are created.

    Args:
        env: Environment with render method.
        clip_data: List of dicts, each containing:
            - "states": Sequence of states
            - "indices": Code indices (optional)
            - "clip_idx": Clip index for label (optional)
        output_path: Base path for output videos (will append _1, _2, etc.).
        max_rows: Maximum rows per grid.
        max_cols: Maximum columns per grid.
        camera: Camera name for rendering.
        cell_width: Width of each cell.
        cell_height: Height of each cell.
        fps: Frames per second.
        num_codes: Total number of codes.
        code_bar_height: Height of code bar per cell.
        padding: Padding between cells.
        bg_color: Background color (RGB).

    Returns:
        List of paths to saved video files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clips_per_grid = max_rows * max_cols
    num_grids = (len(clip_data) + clips_per_grid - 1) // clips_per_grid

    # Get colormap
    if num_codes is None:
        all_indices = []
        for clip in clip_data:
            if "indices" in clip and clip["indices"] is not None:
                all_indices.extend(clip["indices"].flatten())
        num_codes = max(all_indices) + 1 if all_indices else 64
    code_colors = get_nature_colormap(num_codes)

    output_paths = []

    for grid_idx in range(num_grids):
        start_idx = grid_idx * clips_per_grid
        end_idx = min(start_idx + clips_per_grid, len(clip_data))
        grid_clips = clip_data[start_idx:end_idx]

        num_clips = len(grid_clips)
        grid_cols = min(num_clips, max_cols)
        grid_rows = (num_clips + grid_cols - 1) // grid_cols

        logging.info(f"  Rendering grid {grid_idx + 1}/{num_grids} "
                     f"({grid_rows}x{grid_cols}, {num_clips} clips)")

        # Pre-render all frames for each clip
        rendered_clips = []
        for clip in grid_clips:
            states = clip["states"]
            frames = env.render(states, camera=camera,
                                height=cell_height - code_bar_height,
                                width=cell_width)

            indices = clip.get("indices")
            clip_idx = clip.get("clip_idx")

            processed = []
            for i, frame in enumerate(frames):
                # Pad frame to full cell height
                full_frame = np.ones((cell_height, cell_width, 3),
                                     dtype=np.uint8) * 255
                full_frame[:cell_height - code_bar_height, :] = frame

                # Add code bar
                if indices is not None:
                    full_frame = add_code_transition_bar(
                        full_frame,
                        current_frame_idx=i,
                        all_indices=indices,
                        code_colors=code_colors,
                        bar_height=code_bar_height,
                        font_size=12,
                        show_playhead=True,
                    )

                # Add clip label
                if clip_idx is not None:
                    full_frame = add_text_overlay(
                        full_frame,
                        f"#{clip_idx}",
                        position=(5, 5),
                        font_size=12,
                        bg_color=(255, 255, 255, 220),
                        text_color=(0, 0, 0),
                        padding=3,
                    )

                processed.append(full_frame)
            rendered_clips.append(processed)

        # Find max frames across clips
        max_frames = max(len(c) for c in rendered_clips)

        # Calculate grid dimensions
        grid_width = grid_cols * cell_width + (grid_cols - 1) * padding
        grid_height = grid_rows * cell_height + (grid_rows - 1) * padding

        # Assemble grid frames
        grid_frames = []
        for frame_idx in range(max_frames):
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8)
            grid[:] = bg_color

            for clip_num, clip_frames in enumerate(rendered_clips):
                row = clip_num // grid_cols
                col = clip_num % grid_cols

                # Use last frame if past end
                frame = clip_frames[min(frame_idx, len(clip_frames) - 1)]

                y_start = row * (cell_height + padding)
                x_start = col * (cell_width + padding)

                grid[y_start:y_start + cell_height,
                     x_start:x_start + cell_width] = frame

            grid_frames.append(grid)

        # Save video
        if num_grids > 1:
            video_path = output_path.parent / f"{output_path.stem}_{grid_idx + 1}{output_path.suffix}"
        else:
            video_path = output_path

        with imageio.get_writer(str(video_path), fps=fps) as writer:
            for frame in grid_frames:
                writer.append_data(frame)

        output_paths.append(str(video_path))
        logging.info(f"  Saved grid video to {video_path}")

    return output_paths


def render_grid_video(
    env: Any,
    rollout_states_list: list[Sequence[Any]],
    labels: list[str],
    output_path: str | Path,
    grid_cols: int = 4,
    camera: str | None = None,
    cell_width: int = 320,
    cell_height: int = 240,
    fps: int = 50,
) -> str:
    """Render multiple rollouts in a grid layout (legacy API).

    Args:
        env: Environment with render method.
        rollout_states_list: List of rollout sequences.
        labels: Label for each rollout.
        output_path: Path to save output video.
        grid_cols: Number of columns in grid.
        camera: Camera name.
        cell_width: Width of each cell.
        cell_height: Height of each cell.
        fps: Frames per second.

    Returns:
        Path to saved video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_rollouts = len(rollout_states_list)
    grid_rows = (num_rollouts + grid_cols - 1) // grid_cols

    all_frames = []
    for states in rollout_states_list:
        frames = env.render(states, camera=camera,
                            height=cell_height, width=cell_width)
        all_frames.append(frames)

    max_frames = max(len(f) for f in all_frames)

    grid_frames = []
    for frame_idx in range(max_frames):
        grid = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3),
                        dtype=np.uint8)

        for rollout_idx, (frames, label) in enumerate(zip(all_frames, labels)):
            row = rollout_idx // grid_cols
            col = rollout_idx % grid_cols

            frame = frames[min(frame_idx, len(frames) - 1)]
            frame = add_text_overlay(frame, label, font_size=14)

            y_start = row * cell_height
            x_start = col * cell_width
            grid[y_start:y_start + cell_height,
                 x_start:x_start + cell_width] = frame

        grid_frames.append(grid)

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in grid_frames:
            writer.append_data(frame)

    return str(output_path)


def render_comparison_video(
    env: Any,
    rollout_states: Sequence[Any],
    reference_states: Sequence[Any],
    output_path: str | Path,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    indices: np.ndarray | None = None,
) -> str:
    """Render side-by-side comparison of rollout and reference.

    Args:
        env: Environment with render method.
        rollout_states: Sequence of rollout states.
        reference_states: Sequence of reference states.
        output_path: Path to save output video.
        camera: Camera name.
        width: Width per panel.
        height: Video height.
        fps: Frames per second.
        indices: Optional codebook indices for overlay.

    Returns:
        Path to saved video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rollout_frames = env.render(rollout_states, camera=camera,
                                height=height, width=width)
    ref_frames = env.render(reference_states, camera=camera,
                            height=height, width=width)

    combined_frames = []
    num_frames = min(len(rollout_frames), len(ref_frames))

    for i in range(num_frames):
        rollout_frame = add_text_overlay(rollout_frames[i], "Rollout",
                                         position=(10, 10))
        ref_frame = add_text_overlay(ref_frames[i], "Reference",
                                     position=(10, 10))

        if indices is not None and i < len(indices):
            rollout_frame = add_text_overlay(
                rollout_frame, f"Code: {int(indices[i])}",
                position=(10, 40)
            )

        combined = np.concatenate([rollout_frame, ref_frame], axis=1)
        combined_frames.append(combined)

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in combined_frames:
            writer.append_data(frame)

    return str(output_path)
