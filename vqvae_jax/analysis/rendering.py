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
    codes_used = [code for code, frames in code_to_frame_indices.items() if len(frames) >= min_frames_per_code]

    logging.info(f"  Creating videos for {len(codes_used)} codes with >= {min_frames_per_code} frames")

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
                    bg_color=(int(code_color[0]), int(code_color[1]), int(code_color[2]), 200),
                    text_color=(255, 255, 255) if sum(code_color) < 384 else (0, 0, 0),
                )

                # Add progress indicator bar at bottom
                bar = np.zeros((code_bar_height, width, 3), dtype=np.uint8)
                bar[:] = code_color
                # Add progress marker
                progress = int((i / max(len(frame_indices) - 1, 1)) * width)
                bar[:, max(0, progress - 2):min(width, progress + 2)] = [255, 255, 255]

                # Combine frame and bar
                combined = np.vstack([frame, bar])
                code_frames.append(combined)

        if code_frames:
            video_path = output_dir / f"code_{code_idx}.mp4"
            with imageio.get_writer(str(video_path), fps=fps) as writer:
                for frame in code_frames:
                    writer.append_data(frame)
            output_paths[code_idx] = str(video_path)
            logging.info(f"    Code {code_idx}: {len(code_frames)} frames -> {video_path.name}")

    return output_paths


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


# =============================================================================
# COMMUNITY-BASED VISUALIZATION
# =============================================================================

# Community color palette (8 distinct, saturated colors)
COMMUNITY_COLORS = np.array([
    [66, 133, 244],    # Blue
    [234, 67, 53],     # Red
    [251, 188, 5],     # Yellow
    [52, 168, 83],     # Green
    [155, 89, 182],    # Purple
    [26, 188, 156],    # Teal
    [241, 196, 15],    # Gold
    [230, 126, 34],    # Orange
], dtype=np.uint8)


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


def add_community_transition_bar(
    frame: np.ndarray,
    current_frame_idx: int,
    all_indices: np.ndarray,
    code_to_community: dict[int, int],
    community_colors: np.ndarray,
    bar_height: int = 40,
    playhead_width: int = 3,
    show_playhead: bool = True,
    show_label: bool = True,
    font_size: int = 16,
) -> np.ndarray:
    """Add a community-colored transition timeline bar.

    Similar to add_code_transition_bar but colors by community assignment
    rather than individual code.

    Args:
        frame: Input frame as numpy array, shape [H, W, 3].
        current_frame_idx: Current frame index in the sequence.
        all_indices: Array of all code indices for the full sequence.
        code_to_community: Dict mapping code index to community ID.
        community_colors: Color for each community, shape [n_communities, 3].
        bar_height: Height of the timeline bar in pixels.
        playhead_width: Width of the playhead marker.
        show_playhead: Whether to show the playhead marker.
        show_label: Whether to show community label.
        font_size: Font size for label.

    Returns:
        Frame with community timeline overlay.
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    num_frames = len(all_indices)
    y_start = h - bar_height

    # Create timeline bar
    timeline = np.ones((bar_height, w, 3), dtype=np.uint8) * 255

    # Draw border at top
    timeline[:1, :] = [200, 200, 200]

    # Draw community segments
    for i, code_idx in enumerate(all_indices):
        x_start = int(i * w / num_frames)
        x_end = int((i + 1) * w / num_frames)
        comm_id = code_to_community.get(int(code_idx), 0)
        color = community_colors[comm_id % len(community_colors)]
        timeline[1:, x_start:x_end] = color

    # Overlay timeline on frame
    result[y_start:y_start + bar_height, :] = timeline

    # Draw playhead
    if show_playhead:
        playhead_x = int(current_frame_idx * w / num_frames)
        playhead_x = min(playhead_x, w - playhead_width)

        if playhead_x > 0:
            result[y_start:, playhead_x - 1:playhead_x] = [50, 50, 50]
        result[y_start:, playhead_x:playhead_x + playhead_width] = [255, 255, 255]
        if playhead_x + playhead_width < w:
            result[y_start:, playhead_x + playhead_width:playhead_x + playhead_width + 1] = [50, 50, 50]

    # Add community label
    if show_label and current_frame_idx < len(all_indices):
        current_code = int(all_indices[current_frame_idx])
        comm_id = code_to_community.get(current_code, 0)
        color = community_colors[comm_id % len(community_colors)]

        label_img = Image.fromarray(result)
        draw = ImageDraw.Draw(label_img)
        font = _get_font(font_size, bold=True)

        label_text = f"C{comm_id} (code {current_code})"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 6
        box_x, box_y = 10, 10

        draw.rounded_rectangle(
            [box_x - padding, box_y - padding,
             box_x + text_width + padding, box_y + text_height + padding],
            radius=4,
            fill=tuple(color)
        )

        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        draw.text((box_x, box_y), label_text, font=font, fill=text_color)

        result = np.array(label_img)

    return result


def render_rollout_with_community_bar(
    env: Any,
    rollout_states: Sequence[Any],
    output_path: str | Path,
    indices: np.ndarray,
    code_to_community: dict[int, int],
    n_communities: int,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    bar_height: int = 40,
    clip_idx: int | None = None,
) -> str:
    """Render rollout with community-colored timeline bar.

    Args:
        env: Environment with render method.
        rollout_states: Sequence of environment states to render.
        output_path: Path to save output video.
        indices: Code indices per frame.
        code_to_community: Dict mapping code to community ID.
        n_communities: Number of communities.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        bar_height: Height of the community bar.
        clip_idx: Optional clip index to display.

    Returns:
        Path to saved video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"  Rendering {len(rollout_states)} frames with community bar...")
    frames = env.render(rollout_states, camera=camera, height=height, width=width)

    community_colors = get_community_colormap(n_communities)

    processed_frames = []
    for i, frame in enumerate(frames):
        if i < len(indices):
            frame = add_community_transition_bar(
                frame,
                current_frame_idx=i,
                all_indices=indices,
                code_to_community=code_to_community,
                community_colors=community_colors,
                bar_height=bar_height,
            )

        # Add clip index if provided
        if clip_idx is not None:
            frame = add_text_overlay(
                frame,
                f"Clip {clip_idx}",
                position=(10, 50),
                font_size=14,
            )

        processed_frames.append(frame)

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in processed_frames:
            writer.append_data(frame)

    logging.info(f"  Saved community rollout video to {output_path}")
    return str(output_path)


def render_community_grid_video(
    env: Any,
    results: list["InferenceResult"],
    structure: "CommunityStructure",
    output_path: str | Path,
    samples_per_community: int = 4,
    camera: str | None = None,
    cell_width: int = 320,
    cell_height: int = 240,
    fps: int = 50,
    max_frames: int = 200,
) -> str:
    """Render grid video with rows for each community, columns for sample clips.

    Args:
        env: Environment with render method.
        results: List of InferenceResult with states.
        structure: CommunityStructure from community analysis.
        output_path: Path to save output video.
        samples_per_community: Number of sample clips per community row.
        camera: Camera name for rendering.
        cell_width: Width of each cell.
        cell_height: Height of each cell (includes bar).
        fps: Frames per second.
        max_frames: Maximum frames to render.

    Returns:
        Path to saved video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_communities = structure.n_communities
    community_colors = get_community_colormap(n_communities)

    # Select representative clips for each community
    # Find clips that spend significant time in each community
    community_clips: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_communities)}

    for result_idx, result in enumerate(results):
        if result.states is None:
            continue

        # Count frames per community in this clip
        comm_frames = {i: 0 for i in range(n_communities)}
        for code_idx in result.code_indices:
            comm_id = structure.code_to_community.get(int(code_idx), 0)
            comm_frames[comm_id] += 1

        # Assign clip to community with most frames
        dominant_comm = max(comm_frames, key=comm_frames.get)
        community_clips[dominant_comm].append((result_idx, comm_frames[dominant_comm]))

    # Sort clips by dominance and select top samples
    selected_clips: list[list[tuple[int, "InferenceResult"]]] = []
    for comm_id in range(n_communities):
        clips = sorted(community_clips[comm_id], key=lambda x: x[1], reverse=True)
        selected = []
        for result_idx, _ in clips[:samples_per_community]:
            selected.append((result_idx, results[result_idx]))
        # Pad with None if not enough clips
        while len(selected) < samples_per_community:
            selected.append((None, None))
        selected_clips.append(selected)

    # Determine grid size
    grid_rows = n_communities
    grid_cols = samples_per_community
    bar_height = 25

    render_height = cell_height - bar_height
    grid_width = grid_cols * cell_width
    grid_height = grid_rows * cell_height

    logging.info(f"  Rendering community grid: {grid_rows}x{grid_cols}")

    # Pre-render all clips
    rendered_clips: list[list[list[np.ndarray] | None]] = []
    for comm_id in range(n_communities):
        comm_renders = []
        for result_idx, result in selected_clips[comm_id]:
            if result is None or result.states is None:
                comm_renders.append(None)
                continue

            # Render this clip
            states = result.states[:max_frames]
            frames = env.render(states, camera=camera, height=render_height, width=cell_width)

            # Add community bar to each frame
            processed = []
            indices = result.code_indices[:len(frames)]
            for i, frame in enumerate(frames):
                # Create cell with bar
                cell = np.ones((cell_height, cell_width, 3), dtype=np.uint8) * 255
                cell[:render_height, :] = frame

                # Add community color bar
                if i < len(indices):
                    code_idx = int(indices[i])
                    c_id = structure.code_to_community.get(code_idx, 0)
                    color = community_colors[c_id]
                    cell[render_height:, :] = color

                    # Add thin playhead
                    playhead_x = int(i * cell_width / len(indices))
                    cell[render_height:, max(0, playhead_x):min(cell_width, playhead_x + 2)] = [255, 255, 255]

                processed.append(cell)

            comm_renders.append(processed)
        rendered_clips.append(comm_renders)

    # Find max frames across all clips
    all_frame_counts = []
    for comm_renders in rendered_clips:
        for clip_frames in comm_renders:
            if clip_frames is not None:
                all_frame_counts.append(len(clip_frames))

    if not all_frame_counts:
        logging.warning("No clips with states found for community grid")
        return ""

    total_frames = min(max(all_frame_counts), max_frames)

    # Create placeholder for empty cells
    empty_cell = np.ones((cell_height, cell_width, 3), dtype=np.uint8) * 220

    # Add community labels to empty cells
    for comm_id in range(n_communities):
        label_img = Image.fromarray(empty_cell.copy())
        draw = ImageDraw.Draw(label_img)
        font = _get_font(24, bold=True)
        text = f"Community {comm_id}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (cell_width - text_width) // 2
        draw.text((x, cell_height // 2 - 12), text, font=font, fill=(100, 100, 100))

    # Assemble grid frames
    grid_frames = []
    for frame_idx in range(total_frames):
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

        for row, comm_renders in enumerate(rendered_clips):
            for col, clip_frames in enumerate(comm_renders):
                y_start = row * cell_height
                x_start = col * cell_width

                if clip_frames is None:
                    grid[y_start:y_start + cell_height, x_start:x_start + cell_width] = empty_cell
                else:
                    # Use last frame if past end
                    fidx = min(frame_idx, len(clip_frames) - 1)
                    grid[y_start:y_start + cell_height, x_start:x_start + cell_width] = clip_frames[fidx]

        # Add row labels (community IDs)
        grid_img = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_img)
        font = _get_font(18, bold=True)

        for row in range(grid_rows):
            y = row * cell_height + 5
            color = tuple(community_colors[row])
            draw.rounded_rectangle([5, y, 35, y + 22], radius=3, fill=color)
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            draw.text((10, y + 2), f"C{row}", font=font, fill=text_color)

        grid_frames.append(np.array(grid_img))

    # Write video
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in grid_frames:
            writer.append_data(frame)

    logging.info(f"  Saved community grid video to {output_path}")
    return str(output_path)


def render_clips_from_qpos(
    env: Any,
    results: list["InferenceResult"],
    output_dir: str | Path,
    num_clips: int = 10,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    num_codes: int = 128,
    code_to_community: dict[int, int] | None = None,
    n_communities: int | None = None,
) -> dict[str, str]:
    """Render clips from qpos with code and community transition bars.

    Args:
        env: Environment with mj_model attribute.
        results: List of InferenceResult with qpos and code_indices.
        output_dir: Directory to save output videos.
        num_clips: Number of clips to render.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        num_codes: Total number of codes.
        code_to_community: Mapping from code index to community ID.
        n_communities: Number of communities.

    Returns:
        Dictionary mapping output names to file paths.
    """
    import mujoco

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get MuJoCo model and data from environment
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Set up renderer
    bar_height = 40
    render_height = height - bar_height
    renderer = mujoco.Renderer(mj_model, height=render_height, width=width)

    # Find camera ID
    camera_id = -1
    if camera is not None:
        for i in range(mj_model.ncam):
            cam_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name and camera in cam_name:
                camera_id = i
                break

    # Get colormaps
    code_colors = get_nature_colormap(num_codes)
    community_colors = None
    if n_communities is not None and n_communities > 0:
        community_colors = get_community_colormap(n_communities)

    paths = {}

    # Select clips to render (first num_clips with valid qpos)
    clips_to_render = []
    for result in results:
        if result.qpos is not None and len(result.qpos) > 0:
            clips_to_render.append(result)
        if len(clips_to_render) >= num_clips:
            break

    logging.info(f"Rendering {len(clips_to_render)} clips from qpos...")

    for clip_num, result in enumerate(clips_to_render):
        clip_idx = result.clip_idx
        qpos_data = result.qpos
        indices = result.code_indices

        logging.info(f"  Clip {clip_num} (idx={clip_idx}): {len(qpos_data)} frames")

        # Render frames from qpos
        raw_frames = []
        for qpos in qpos_data:
            mj_data.qpos[:] = qpos[:mj_model.nq]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera_id)
            frame = renderer.render()
            raw_frames.append(frame.copy())

        # === Render with code bar ===
        code_frames = []
        for i, frame in enumerate(raw_frames):
            # Create full frame with bar
            full_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            full_frame[:render_height, :] = frame

            # Add code transition bar
            full_frame = add_code_transition_bar(
                full_frame,
                current_frame_idx=i,
                all_indices=indices,
                code_colors=code_colors,
                bar_height=bar_height,
            )

            # Add clip label
            full_frame = add_text_overlay(
                full_frame,
                f"Clip {clip_idx}",
                position=(10, 10),
                font_size=18,
            )

            code_frames.append(full_frame)

        # Save code video
        code_path = output_dir / f"clip_{clip_idx:03d}_codes.mp4"
        with imageio.get_writer(str(code_path), fps=fps) as writer:
            for frame in code_frames:
                writer.append_data(frame)
        paths[f"clip_{clip_idx}_codes"] = str(code_path)
        logging.info(f"    Saved {code_path}")

        # === Render with community bar ===
        if code_to_community is not None and community_colors is not None:
            comm_frames = []
            for i, frame in enumerate(raw_frames):
                # Create full frame with bar
                full_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                full_frame[:render_height, :] = frame

                # Add community transition bar
                full_frame = add_community_transition_bar(
                    full_frame,
                    current_frame_idx=i,
                    all_indices=indices,
                    code_to_community=code_to_community,
                    community_colors=community_colors,
                    bar_height=bar_height,
                )

                # Add clip label
                full_frame = add_text_overlay(
                    full_frame,
                    f"Clip {clip_idx}",
                    position=(10, 10),
                    font_size=18,
                )

                comm_frames.append(full_frame)

            # Save community video
            comm_path = output_dir / f"clip_{clip_idx:03d}_communities.mp4"
            with imageio.get_writer(str(comm_path), fps=fps) as writer:
                for frame in comm_frames:
                    writer.append_data(frame)
            paths[f"clip_{clip_idx}_communities"] = str(comm_path)
            logging.info(f"    Saved {comm_path}")

    renderer.close()
    return paths


def render_popular_code_videos(
    env: Any,
    results: list["InferenceResult"],
    popular_codes: list[tuple[int, int]],
    segments_by_code: dict[int, list[Any]],
    output_dir: str | Path,
    num_codes: int,
    max_segments_per_code: int = 6,
    min_segment_frames: int = 10,
    cell_width: int = 200,
    cell_height: int = 180,
    camera: str | None = None,
    fps: int = 50,
) -> dict[int, str]:
    """Render videos for most popular codes showing multiple segments side-by-side.

    Creates grid videos where each column shows a different segment of the same code,
    arranged by duration (longest segments first).

    Args:
        env: Environment with mj_model attribute for rendering.
        results: List of InferenceResult with qpos and code_indices.
        popular_codes: List of (code_idx, frame_count) tuples for popular codes.
        segments_by_code: Dictionary mapping code_idx to list of CodeSegment.
        output_dir: Directory to save videos.
        num_codes: Total number of codes in codebook.
        max_segments_per_code: Maximum segments to show per code video.
        min_segment_frames: Minimum frames for segment inclusion.
        cell_width: Width of each segment cell.
        cell_height: Height of each segment cell (including code bar).
        camera: Camera name for rendering.
        fps: Frames per second.

    Returns:
        Dictionary mapping code_idx to video path.
    """
    import mujoco

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a map from clip_idx to result for quick lookup
    results_by_clip = {r.clip_idx: r for r in results}

    # Get MuJoCo model and data from environment
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Set up renderer
    code_bar_height = 30
    render_height = cell_height - code_bar_height
    renderer = mujoco.Renderer(mj_model, height=render_height, width=cell_width)

    # Find camera ID
    camera_id = -1
    if camera is not None:
        for i in range(mj_model.ncam):
            cam_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name and camera in cam_name:
                camera_id = i
                break

    code_colors = get_nature_colormap(num_codes)
    output_paths: dict[int, str] = {}

    for code_idx, frame_count in popular_codes:
        segments = segments_by_code.get(code_idx, [])

        # Filter by minimum frames
        segments = [s for s in segments if s.duration >= min_segment_frames]
        if len(segments) == 0:
            logging.info(f"Code {code_idx}: no segments with >= {min_segment_frames} frames")
            continue

        # Sort by duration descending, take top N
        segments = sorted(segments, key=lambda s: s.duration, reverse=True)
        segments = segments[:max_segments_per_code]

        logging.info(
            f"Code {code_idx} ({frame_count} total frames): rendering {len(segments)} segments "
            f"(max duration: {segments[0].duration})"
        )

        # Find max duration for this code's segments
        max_duration = max(s.duration for s in segments)
        num_cols = len(segments)

        # Pre-render all segment frames from qpos
        segment_frames: list[list[np.ndarray]] = []

        for seg in segments:
            result = results_by_clip.get(seg.clip_idx)
            if result is None or result.qpos is None:
                logging.warning(
                    f"Missing qpos for clip {seg.clip_idx}, skipping segment"
                )
                segment_frames.append([])
                continue

            # Extract qpos for this segment
            segment_qpos = result.qpos[seg.start_frame : seg.end_frame]

            # Render frames from qpos
            frames = []
            for qpos in segment_qpos:
                mj_data.qpos[:] = qpos[: mj_model.nq]
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=camera_id)
                frame = renderer.render()
                frames.append(frame.copy())

            # Add segment label overlay to first frame
            if len(frames) > 0:
                label = f"clip {seg.clip_idx} | {seg.duration}f"
                frames[0] = add_text_overlay(
                    frames[0],
                    label,
                    position=(5, 5),
                    font_size=10,
                    bg_color=(0, 0, 0, 180),
                    text_color=(255, 255, 255),
                    padding=3,
                )

            segment_frames.append(frames)

        if all(len(f) == 0 for f in segment_frames):
            logging.warning(f"No valid segments rendered for code {code_idx}")
            continue

        # Assemble grid video
        grid_width = num_cols * cell_width + (num_cols - 1) * 2
        grid_height = cell_height

        video_frames = []
        for frame_idx in range(max_duration):
            # Create grid frame with white background
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            for col, frames in enumerate(segment_frames):
                if len(frames) == 0:
                    continue

                # Use last frame if past end
                f_idx = min(frame_idx, len(frames) - 1)
                frame = frames[f_idx]

                # Create cell with padded content and code bar
                cell = np.ones((cell_height, cell_width, 3), dtype=np.uint8) * 255

                # Copy rendered frame to top portion
                h, w = frame.shape[:2]
                cell[:min(h, render_height), :min(w, cell_width)] = frame[:min(h, render_height), :min(w, cell_width)]

                # Add code bar at bottom
                bar = np.zeros((code_bar_height, cell_width, 3), dtype=np.uint8)
                bar[:] = code_colors[code_idx]

                # Add progress indicator
                total_frames = len(frames)
                progress = int((f_idx / max(total_frames - 1, 1)) * cell_width)
                bar[:, max(0, progress - 1) : min(cell_width, progress + 2)] = 255

                cell[render_height:, :] = bar

                x_start = col * (cell_width + 2)
                grid[:, x_start : x_start + cell_width] = cell

            video_frames.append(grid)

        # Write video
        video_path = output_dir / f"code_{code_idx}_popular.mp4"
        with imageio.get_writer(str(video_path), fps=fps) as writer:
            for frame in video_frames:
                writer.append_data(frame)

        output_paths[code_idx] = str(video_path)
        logging.info(f"  Saved {video_path.name}")

    renderer.close()
    return output_paths
