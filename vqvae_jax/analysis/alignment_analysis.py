"""DTW alignment analysis for VQ-VAE code segments.

This module provides functions for aligning code segments using
Dynamic Time Warping (DTW) and rendering aligned comparisons.
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
from .segment_analysis import CodeSegment, extract_code_segments
from .rendering import get_nature_colormap, add_text_overlay


@dataclass
class AlignmentInfo:
    """Information about DTW alignment between two segments.

    Attributes:
        ref_clip_idx: Clip index of the reference segment.
        ref_start: Start frame of reference segment.
        ref_end: End frame of reference segment.
        query_clip_idx: Clip index of the query segment.
        query_start: Start frame of query segment.
        query_end: End frame of query segment.
        dtw_distance: DTW distance between the segments.
        path: DTW alignment path as list of (ref_idx, query_idx) tuples.
    """

    ref_clip_idx: int
    ref_start: int
    ref_end: int
    query_clip_idx: int
    query_start: int
    query_end: int
    dtw_distance: float
    path: list[tuple[int, int]]


def compute_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    return_path: bool = False,
) -> float | tuple[float, list[tuple[int, int]]]:
    """Compute DTW distance between two sequences.

    Uses a simple DTW implementation without external dependencies.
    For better performance with large sequences, consider fastdtw.

    Args:
        seq1: First sequence, shape [T1, D].
        seq2: Second sequence, shape [T2, D].
        return_path: Whether to return the alignment path.

    Returns:
        DTW distance, or (distance, path) if return_path is True.
    """
    n, m = len(seq1), len(seq2)

    # Compute pairwise distances
    distances = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distances[i, j] = np.linalg.norm(seq1[i] - seq2[j])

    # DTW dynamic programming
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = distances[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    if not return_path:
        return dtw[n, m]

    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dtw[i - 1, j - 1], (i - 1, j - 1)),
            (dtw[i - 1, j], (i - 1, j)),
            (dtw[i, j - 1], (i, j - 1)),
        ]
        _, (i, j) = min(candidates, key=lambda x: x[0])

    path.reverse()
    return dtw[n, m], path


def find_longest_segment(segments: list[CodeSegment]) -> CodeSegment | None:
    """Find the longest segment from a list.

    Args:
        segments: List of CodeSegment objects.

    Returns:
        The longest segment, or None if list is empty.
    """
    if not segments:
        return None
    return max(segments, key=lambda s: s.duration)


def align_segments_to_reference(
    results: Sequence[InferenceResult],
    ref_segment: CodeSegment,
    query_segments: list[CodeSegment],
    feature: str = "qpos",
) -> list[AlignmentInfo]:
    """Align query segments to a reference segment using DTW.

    Args:
        results: List of InferenceResult with qpos/qvel.
        ref_segment: The reference segment to align to.
        query_segments: List of segments to align.
        feature: Feature to use for alignment ("qpos" or "qvel").

    Returns:
        List of AlignmentInfo for each query segment.
    """
    # Build lookup for results
    results_by_clip = {r.clip_idx: r for r in results}

    # Get reference features
    ref_result = results_by_clip.get(ref_segment.clip_idx)
    if ref_result is None:
        logging.warning(f"Missing result for reference clip {ref_segment.clip_idx}")
        return []

    ref_data = getattr(ref_result, feature, None)
    if ref_data is None:
        logging.warning(f"Missing {feature} for reference clip {ref_segment.clip_idx}")
        return []

    ref_features = ref_data[ref_segment.start_frame : ref_segment.end_frame]

    alignments = []
    for query_seg in query_segments:
        # Skip same segment
        if (
            query_seg.clip_idx == ref_segment.clip_idx
            and query_seg.start_frame == ref_segment.start_frame
        ):
            continue

        query_result = results_by_clip.get(query_seg.clip_idx)
        if query_result is None:
            continue

        query_data = getattr(query_result, feature, None)
        if query_data is None:
            continue

        query_features = query_data[query_seg.start_frame : query_seg.end_frame]

        # Compute DTW
        distance, path = compute_dtw_distance(
            ref_features, query_features, return_path=True
        )

        alignments.append(
            AlignmentInfo(
                ref_clip_idx=ref_segment.clip_idx,
                ref_start=ref_segment.start_frame,
                ref_end=ref_segment.end_frame,
                query_clip_idx=query_seg.clip_idx,
                query_start=query_seg.start_frame,
                query_end=query_seg.end_frame,
                dtw_distance=float(distance),
                path=path,
            )
        )

    # Sort by distance
    alignments.sort(key=lambda x: x.dtw_distance)
    return alignments


def warp_frames_to_reference(
    frames: list[np.ndarray],
    path: list[tuple[int, int]],
    ref_length: int,
) -> list[np.ndarray]:
    """Warp frames according to DTW alignment path.

    Args:
        frames: List of frames to warp.
        path: DTW path as list of (ref_idx, query_idx).
        ref_length: Length of reference sequence.

    Returns:
        List of warped frames matching reference length.
    """
    warped = []
    current_ref_idx = 0

    for ref_idx, query_idx in path:
        # Fill any gaps in reference indices
        while current_ref_idx < ref_idx:
            # Repeat previous frame
            warped.append(frames[query_idx] if warped else frames[0])
            current_ref_idx += 1

        if query_idx < len(frames):
            warped.append(frames[query_idx])
            current_ref_idx += 1

    # Fill remaining frames if needed
    while len(warped) < ref_length:
        warped.append(warped[-1] if warped else frames[-1])

    return warped[:ref_length]


def render_aligned_row(
    env: Any,
    results: Sequence[InferenceResult],
    code_idx: int,
    ref_segment: CodeSegment,
    alignments: list[AlignmentInfo],
    output_path: str | Path,
    max_segments: int = 5,
    cell_width: int = 200,
    cell_height: int = 150,
    camera: str | None = None,
    fps: int = 50,
    num_codes: int = 64,
) -> str:
    """Render a row of aligned segments for a single code.

    Creates a video with the reference segment followed by aligned segments,
    all time-warped to match the reference length.

    Args:
        env: Environment with render method.
        results: List of InferenceResult with states.
        code_idx: The code index.
        ref_segment: Reference segment (longest).
        alignments: List of AlignmentInfo from align_segments_to_reference.
        output_path: Path to save the video.
        max_segments: Maximum number of segments to include (including reference).
        cell_width: Width of each cell.
        cell_height: Height of each cell.
        camera: Camera name for rendering.
        fps: Frames per second.
        num_codes: Total number of codes for coloring.

    Returns:
        Path to saved video.
    """
    import imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_by_clip = {r.clip_idx: r for r in results}
    code_colors = get_nature_colormap(num_codes)

    # Get reference result
    ref_result = results_by_clip.get(ref_segment.clip_idx)
    if ref_result is None or ref_result.states is None:
        logging.warning(f"Missing states for reference clip {ref_segment.clip_idx}")
        return ""

    # Render reference frames
    ref_states = ref_result.states[ref_segment.start_frame : ref_segment.end_frame]
    ref_frames = env.render(
        ref_states, camera=camera, height=cell_height - 30, width=cell_width
    )

    # Add label to reference
    ref_frames[0] = add_text_overlay(
        ref_frames[0],
        f"REF clip {ref_segment.clip_idx} | {ref_segment.duration}f",
        position=(5, 5),
        font_size=10,
        bg_color=(0, 0, 0, 200),
        text_color=(255, 255, 0),  # Yellow for reference
        padding=3,
    )

    ref_length = len(ref_frames)

    # Collect aligned segment frames
    all_frames = [ref_frames]

    for align_info in alignments[: max_segments - 1]:
        query_result = results_by_clip.get(align_info.query_clip_idx)
        if query_result is None or query_result.states is None:
            continue

        # Render query frames
        query_states = query_result.states[align_info.query_start : align_info.query_end]
        query_frames = env.render(
            query_states, camera=camera, height=cell_height - 30, width=cell_width
        )

        # Warp to reference length
        warped_frames = warp_frames_to_reference(
            query_frames, align_info.path, ref_length
        )

        # Add label
        warped_frames[0] = add_text_overlay(
            warped_frames[0],
            f"clip {align_info.query_clip_idx} | d={align_info.dtw_distance:.1f}",
            position=(5, 5),
            font_size=10,
            bg_color=(0, 0, 0, 180),
            text_color=(255, 255, 255),
            padding=3,
        )

        all_frames.append(warped_frames)

    num_cols = len(all_frames)
    if num_cols == 0:
        return ""

    # Assemble video
    code_bar_height = 30
    grid_width = num_cols * cell_width + (num_cols - 1) * 2
    grid_height = cell_height

    video_frames = []
    for frame_idx in range(ref_length):
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

        for col, frames in enumerate(all_frames):
            if len(frames) == 0:
                continue

            f_idx = min(frame_idx, len(frames) - 1)
            frame = frames[f_idx]

            # Pad to cell size
            padded = np.ones(
                (cell_height - code_bar_height, cell_width, 3), dtype=np.uint8
            ) * 255
            h, w = frame.shape[:2]
            padded[:h, :w] = frame

            # Add code bar
            bar = np.zeros((code_bar_height, cell_width, 3), dtype=np.uint8)
            bar[:] = code_colors[code_idx]

            # Add progress indicator
            progress = int((frame_idx / max(ref_length - 1, 1)) * cell_width)
            bar[:, max(0, progress - 1) : min(cell_width, progress + 2)] = 255

            cell = np.vstack([padded, bar])

            x_start = col * (cell_width + 2)
            grid[:, x_start : x_start + cell_width] = cell

        video_frames.append(grid)

    # Write video
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    logging.info(f"Saved aligned row video to {output_path}")
    return str(output_path)


def save_alignment_analysis(
    output_dir: str | Path,
    alignments_by_code: dict[int, list[AlignmentInfo]],
    ref_segments: dict[int, CodeSegment],
) -> dict[str, str]:
    """Save alignment analysis results.

    Args:
        output_dir: Directory to save outputs.
        alignments_by_code: Dictionary mapping code_idx to alignments.
        ref_segments: Dictionary mapping code_idx to reference segment.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save alignment info
    data = {}
    for code_idx, alignments in alignments_by_code.items():
        ref_seg = ref_segments.get(code_idx)
        code_data = {
            "reference": asdict(ref_seg) if ref_seg else None,
            "alignments": [
                {
                    **asdict(a),
                    "path": None,  # Don't save path (too large)
                }
                for a in alignments
            ],
        }
        data[str(code_idx)] = code_data

    with open(output_dir / "alignment_info.json", "w") as f:
        json.dump(data, f, indent=2)
    paths["alignment_info"] = str(output_dir / "alignment_info.json")

    logging.info(f"Saved alignment analysis to {output_dir}")
    return paths


def run_alignment_analysis(
    env: Any,
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    min_segment_length: int = 10,
    max_pairs_per_code: int = 5,
    dtw_feature: str = "qpos",
    render_videos: bool = True,
    camera: str | None = None,
    fps: int = 50,
) -> dict[str, Any]:
    """Run complete alignment analysis pipeline.

    Args:
        env: Environment for rendering.
        results: List of InferenceResult.
        num_codes: Total number of codes.
        output_dir: Directory to save outputs.
        min_segment_length: Minimum frames for segment inclusion.
        max_pairs_per_code: Maximum aligned segments per code.
        dtw_feature: Feature to use for DTW ("qpos" or "qvel").
        render_videos: Whether to render alignment videos.
        camera: Camera name for rendering.
        fps: Frames per second.

    Returns:
        Dictionary with analysis results and file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Extracting segments for alignment...")
    segments_by_code = extract_code_segments(results, min_segment_frames=min_segment_length)

    alignments_by_code: dict[int, list[AlignmentInfo]] = {}
    ref_segments: dict[int, CodeSegment] = {}
    video_paths: dict[int, str] = {}

    for code_idx in sorted(segments_by_code.keys()):
        segments = segments_by_code[code_idx]

        if len(segments) < 2:
            logging.info(f"Code {code_idx}: skipping (only {len(segments)} segments)")
            continue

        # Find longest segment as reference
        ref_segment = find_longest_segment(segments)
        if ref_segment is None:
            continue

        ref_segments[code_idx] = ref_segment
        logging.info(
            f"Code {code_idx}: aligning {len(segments)} segments "
            f"(ref duration: {ref_segment.duration})"
        )

        # Align other segments to reference
        alignments = align_segments_to_reference(
            results, ref_segment, segments, feature=dtw_feature
        )

        alignments_by_code[code_idx] = alignments[: max_pairs_per_code - 1]

        # Render video if enabled and states available
        if render_videos:
            has_states = any(r.states is not None for r in results)
            if has_states:
                video_path = output_dir / f"code_{code_idx}_aligned_row.mp4"
                path = render_aligned_row(
                    env=env,
                    results=results,
                    code_idx=code_idx,
                    ref_segment=ref_segment,
                    alignments=alignments,
                    output_path=video_path,
                    max_segments=max_pairs_per_code,
                    camera=camera,
                    fps=fps,
                    num_codes=num_codes,
                )
                if path:
                    video_paths[code_idx] = path

    # Save results
    paths = save_alignment_analysis(output_dir, alignments_by_code, ref_segments)

    if video_paths:
        paths["videos"] = video_paths

    return {
        "alignments_by_code": alignments_by_code,
        "ref_segments": ref_segments,
        "paths": paths,
    }
