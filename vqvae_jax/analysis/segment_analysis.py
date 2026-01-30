"""Segment analysis for VQ-VAE codes.

This module provides functions for extracting contiguous code segments,
computing duration statistics, and rendering segment visualization videos.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult
from .rendering import get_nature_colormap, add_code_transition_bar, add_text_overlay


@dataclass
class CodeSegment:
    """A contiguous segment where a single code is active.

    Attributes:
        clip_idx: Index of the reference clip.
        code_idx: The active code index.
        start_frame: First frame of the segment (inclusive).
        end_frame: Last frame of the segment (exclusive).
        duration: Number of frames in the segment.
    """

    clip_idx: int
    code_idx: int
    start_frame: int
    end_frame: int
    duration: int


def extract_code_segments(
    results: Sequence[InferenceResult],
    min_segment_frames: int = 1,
) -> dict[int, list[CodeSegment]]:
    """Extract contiguous segments for each code across all clips.

    Args:
        results: List of InferenceResult with code_indices.
        min_segment_frames: Minimum frames for a segment to be included.

    Returns:
        Dictionary mapping code_idx to list of CodeSegment objects.
    """
    segments_by_code: dict[int, list[CodeSegment]] = {}

    for result in results:
        clip_idx = result.clip_idx
        indices = result.code_indices

        if len(indices) == 0:
            continue

        # Find contiguous segments using run-length encoding
        current_code = int(indices[0])
        segment_start = 0

        for i in range(1, len(indices)):
            if indices[i] != current_code:
                # End of segment
                duration = i - segment_start
                if duration >= min_segment_frames:
                    segment = CodeSegment(
                        clip_idx=clip_idx,
                        code_idx=current_code,
                        start_frame=segment_start,
                        end_frame=i,
                        duration=duration,
                    )
                    if current_code not in segments_by_code:
                        segments_by_code[current_code] = []
                    segments_by_code[current_code].append(segment)

                # Start new segment
                current_code = int(indices[i])
                segment_start = i

        # Handle final segment
        duration = len(indices) - segment_start
        if duration >= min_segment_frames:
            segment = CodeSegment(
                clip_idx=clip_idx,
                code_idx=current_code,
                start_frame=segment_start,
                end_frame=len(indices),
                duration=duration,
            )
            if current_code not in segments_by_code:
                segments_by_code[current_code] = []
            segments_by_code[current_code].append(segment)

    return segments_by_code


@dataclass
class DurationStatistics:
    """Duration statistics for a single code.

    Attributes:
        code_idx: The code index.
        count: Number of segments.
        mean: Mean duration in frames.
        std: Standard deviation of duration.
        median: Median duration.
        min: Minimum duration.
        max: Maximum duration.
    """

    code_idx: int
    count: int
    mean: float
    std: float
    median: float
    min: int
    max: int


def compute_duration_statistics(
    segments_by_code: dict[int, list[CodeSegment]],
    num_codes: int,
) -> list[DurationStatistics]:
    """Compute duration statistics for each code.

    Args:
        segments_by_code: Dictionary mapping code_idx to segments.
        num_codes: Total number of codes.

    Returns:
        List of DurationStatistics for each code.
    """
    stats = []

    for code_idx in range(num_codes):
        segments = segments_by_code.get(code_idx, [])

        if len(segments) == 0:
            stats.append(
                DurationStatistics(
                    code_idx=code_idx,
                    count=0,
                    mean=0.0,
                    std=0.0,
                    median=0.0,
                    min=0,
                    max=0,
                )
            )
        else:
            durations = [s.duration for s in segments]
            stats.append(
                DurationStatistics(
                    code_idx=code_idx,
                    count=len(durations),
                    mean=float(np.mean(durations)),
                    std=float(np.std(durations)),
                    median=float(np.median(durations)),
                    min=int(np.min(durations)),
                    max=int(np.max(durations)),
                )
            )

    return stats


def compute_code_popularity(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> dict[int, int]:
    """Compute total frame count for each code across all clips.

    Args:
        results: List of InferenceResult with code_indices.
        num_codes: Total number of codes in the codebook.

    Returns:
        Dictionary mapping code_idx to total frame count.
    """
    frame_counts: dict[int, int] = {i: 0 for i in range(num_codes)}
    for result in results:
        for code_idx in result.code_indices:
            frame_counts[int(code_idx)] += 1
    return frame_counts


def get_top_k_popular_codes(
    frame_counts: dict[int, int],
    k: int,
    min_frames: int = 50,
) -> list[tuple[int, int]]:
    """Get top K codes by frame count that meet minimum threshold.

    Args:
        frame_counts: Dictionary mapping code_idx to frame count.
        k: Number of top codes to return.
        min_frames: Minimum frame count for a code to be included.

    Returns:
        List of (code_idx, frame_count) tuples sorted by count descending.
    """
    filtered = [
        (code, count) for code, count in frame_counts.items() if count >= min_frames
    ]
    sorted_codes = sorted(filtered, key=lambda x: x[1], reverse=True)
    return sorted_codes[:k]


def plot_duration_distributions(
    stats: list[DurationStatistics],
    segments_by_code: dict[int, list[CodeSegment]],
    output_path: str | Path,
    figsize: tuple[int, int] = (14, 6),
) -> str:
    """Plot duration statistics for all codes.

    Args:
        stats: List of DurationStatistics.
        segments_by_code: Dictionary mapping code_idx to segments.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean duration by code
    codes = [s.code_idx for s in stats]
    means = [s.mean for s in stats]
    stds = [s.std for s in stats]

    ax = axes[0]
    bars = ax.bar(codes, means, yerr=stds, capsize=3, alpha=0.7, color="steelblue")
    ax.set_xlabel("Code Index")
    ax.set_ylabel("Duration (frames)")
    ax.set_title("Mean Duration by Code")
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot of durations
    ax = axes[1]
    durations_per_code = []
    labels = []
    for code_idx in sorted(segments_by_code.keys()):
        segments = segments_by_code[code_idx]
        if len(segments) > 0:
            durations_per_code.append([s.duration for s in segments])
            labels.append(str(code_idx))

    if durations_per_code:
        bp = ax.boxplot(durations_per_code, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xlabel("Code Index")
        ax.set_ylabel("Duration (frames)")
        ax.set_title("Duration Distribution by Code")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def render_code_segment_videos(
    env: Any,
    results: Sequence[InferenceResult],
    segments_by_code: dict[int, list[CodeSegment]],
    output_dir: str | Path,
    num_codes: int,
    max_segments_per_code: int = 10,
    min_segment_frames: int = 10,
    cell_width: int = 200,
    cell_height: int = 150,
    camera: str | None = None,
    fps: int = 50,
) -> dict[int, str]:
    """Render separate videos for each code showing its segments as columns.

    Each code gets its own video with multiple segments arranged horizontally.
    Each segment cell is labeled with clip_idx and duration.

    Args:
        env: Environment with render method.
        results: List of InferenceResult with states.
        segments_by_code: Dictionary mapping code_idx to segments.
        output_dir: Directory to save videos.
        num_codes: Total number of codes.
        max_segments_per_code: Maximum segments to include per code video.
        min_segment_frames: Minimum frames for segment to be included.
        cell_width: Width of each segment cell.
        cell_height: Height of each segment cell (excluding code bar).
        camera: Camera name for rendering.
        fps: Frames per second.

    Returns:
        Dictionary mapping code_idx to video path.
    """
    import imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a map from clip_idx to result for quick lookup
    results_by_clip = {r.clip_idx: r for r in results}

    code_colors = get_nature_colormap(num_codes)
    output_paths: dict[int, str] = {}

    for code_idx in sorted(segments_by_code.keys()):
        segments = segments_by_code[code_idx]

        # Filter by minimum frames
        segments = [s for s in segments if s.duration >= min_segment_frames]
        if len(segments) == 0:
            continue

        # Sort by duration descending, take top N
        segments = sorted(segments, key=lambda s: s.duration, reverse=True)
        segments = segments[:max_segments_per_code]

        logging.info(
            f"Code {code_idx}: rendering {len(segments)} segments "
            f"(max duration: {segments[0].duration})"
        )

        # Find max duration for this code's segments
        max_duration = max(s.duration for s in segments)
        num_cols = len(segments)

        # Pre-render all segment frames
        segment_frames: list[list[np.ndarray]] = []

        for seg in segments:
            result = results_by_clip.get(seg.clip_idx)
            if result is None or result.states is None:
                logging.warning(
                    f"Missing states for clip {seg.clip_idx}, skipping segment"
                )
                segment_frames.append([])
                continue

            # Extract states for this segment
            segment_states = result.states[seg.start_frame : seg.end_frame]

            # Render frames
            frames = env.render(
                segment_states, camera=camera, height=cell_height - 30, width=cell_width
            )

            # Add segment label overlay to first frame
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
            logging.warning(f"No valid segments for code {code_idx}")
            continue

        # Assemble grid video
        code_bar_height = 30
        grid_width = num_cols * cell_width + (num_cols - 1) * 2
        grid_height = cell_height

        video_frames = []
        for frame_idx in range(max_duration):
            # Create grid frame
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            for col, frames in enumerate(segment_frames):
                if len(frames) == 0:
                    continue

                # Use last frame if past end
                f_idx = min(frame_idx, len(frames) - 1)
                frame = frames[f_idx]

                # Pad to cell height minus code bar
                padded = np.ones((cell_height - code_bar_height, cell_width, 3),
                                 dtype=np.uint8) * 255
                h, w = frame.shape[:2]
                padded[:h, :w] = frame

                # Add code bar at bottom
                bar = np.zeros((code_bar_height, cell_width, 3), dtype=np.uint8)
                bar[:] = code_colors[code_idx]

                # Add progress indicator
                total_frames = len(frames)
                progress = int((f_idx / max(total_frames - 1, 1)) * cell_width)
                bar[:, max(0, progress - 1) : min(cell_width, progress + 2)] = 255

                cell = np.vstack([padded, bar])

                x_start = col * (cell_width + 2)
                grid[:, x_start : x_start + cell_width] = cell

            video_frames.append(grid)

        # Write video
        video_path = output_dir / f"code_{code_idx}_segments.mp4"
        with imageio.get_writer(str(video_path), fps=fps) as writer:
            for frame in video_frames:
                writer.append_data(frame)

        output_paths[code_idx] = str(video_path)
        logging.info(f"  Saved {video_path.name}")

    return output_paths


def save_segment_analysis(
    output_dir: str | Path,
    segments_by_code: dict[int, list[CodeSegment]],
    stats: list[DurationStatistics],
) -> dict[str, str]:
    """Save segment analysis results.

    Args:
        output_dir: Directory to save outputs.
        segments_by_code: Dictionary mapping code_idx to segments.
        stats: Duration statistics for each code.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save segment index
    segment_data = {}
    for code_idx, segments in segments_by_code.items():
        segment_data[str(code_idx)] = [asdict(s) for s in segments]

    with open(output_dir / "segment_index.json", "w") as f:
        json.dump(segment_data, f, indent=2)
    paths["segments"] = str(output_dir / "segment_index.json")

    # Save duration stats
    stats_data = [asdict(s) for s in stats]
    with open(output_dir / "duration_stats.json", "w") as f:
        json.dump(stats_data, f, indent=2)
    paths["stats"] = str(output_dir / "duration_stats.json")

    logging.info(f"Saved segment analysis to {output_dir}")
    return paths


def run_segment_analysis(
    env: Any,
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    min_segment_frames: int = 10,
    max_segments_per_code: int = 10,
    render_videos: bool = True,
    camera: str | None = None,
    fps: int = 50,
) -> dict[str, Any]:
    """Run complete segment analysis pipeline.

    Args:
        env: Environment for rendering.
        results: List of InferenceResult.
        num_codes: Total number of codes.
        output_dir: Directory to save outputs.
        min_segment_frames: Minimum frames for segment inclusion.
        max_segments_per_code: Maximum segments per code video.
        render_videos: Whether to render segment videos.
        camera: Camera name for rendering.
        fps: Frames per second for videos.

    Returns:
        Dictionary with analysis results and file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Extracting code segments...")
    segments_by_code = extract_code_segments(results, min_segment_frames=1)

    total_segments = sum(len(s) for s in segments_by_code.values())
    logging.info(
        f"Found {total_segments} segments across {len(segments_by_code)} codes"
    )

    logging.info("Computing duration statistics...")
    stats = compute_duration_statistics(segments_by_code, num_codes)

    # Save data
    paths = save_segment_analysis(output_dir, segments_by_code, stats)

    # Plot duration distributions
    plot_path = plot_duration_distributions(
        stats, segments_by_code, output_dir / "duration_distributions.png"
    )
    paths["duration_plot"] = plot_path

    # Render videos if enabled and states are available
    if render_videos:
        has_states = any(r.states is not None for r in results)
        if has_states:
            logging.info("Rendering segment videos...")
            video_paths = render_code_segment_videos(
                env=env,
                results=results,
                segments_by_code=segments_by_code,
                output_dir=output_dir,
                num_codes=num_codes,
                max_segments_per_code=max_segments_per_code,
                min_segment_frames=min_segment_frames,
                camera=camera,
                fps=fps,
            )
            paths["videos"] = video_paths
        else:
            logging.warning("No states available for rendering segment videos")

    return {
        "segments_by_code": segments_by_code,
        "stats": stats,
        "paths": paths,
    }
