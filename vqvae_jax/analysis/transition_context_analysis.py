"""Transition Context Analysis for VQ-VAE Codes.

This module analyzes whether the same code serves the same function across
different clips by comparing transition contexts (predecessor/successor patterns).

For top K most frequently used codes:
- Compare predecessor distributions across clips
- Compare successor distributions across clips
- Render transition sequences (predecessor → code → successor) from different clips
- Generate interactive HTML viewer for cross-clip comparison
"""

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult
from .rendering import get_nature_colormap


@dataclass
class CodeTransitionContext:
    """Transition context for a single code in a single clip.

    Attributes:
        code_idx: The code index being analyzed.
        clip_idx: The clip this context is from.
        predecessor_dist: Distribution over predecessor codes.
        successor_dist: Distribution over successor codes.
        occurrence_count: Number of times this code appears in the clip.
        predecessor_counts: Raw counts of each predecessor.
        successor_counts: Raw counts of each successor.
    """

    code_idx: int
    clip_idx: int
    predecessor_dist: np.ndarray
    successor_dist: np.ndarray
    occurrence_count: int
    predecessor_counts: np.ndarray
    successor_counts: np.ndarray


@dataclass
class TransitionSegment:
    """A segment showing predecessor → code → successor transition.

    Attributes:
        code_idx: The central code.
        clip_idx: Which clip this segment is from.
        start_frame: Start frame in the clip.
        end_frame: End frame in the clip.
        predecessor_code: The predecessor code.
        successor_code: The successor code.
        code_indices: Full code sequence for this segment.
    """

    code_idx: int
    clip_idx: int
    start_frame: int
    end_frame: int
    predecessor_code: int
    successor_code: int
    code_indices: np.ndarray


@dataclass
class MatchedFramePair:
    """A pair of frames from two clips matched by qpos similarity.

    Attributes:
        frame_i: Frame index in clip i.
        frame_j: Frame index in clip j.
        qpos_distance: Mean absolute qpos difference.
        succ_i: Successor code in clip i.
        succ_j: Successor code in clip j.
    """

    frame_i: int
    frame_j: int
    qpos_distance: float
    succ_i: int
    succ_j: int


@dataclass
class ConditionalTransitionContext:
    """Transition context conditioned on qpos similarity.

    Attributes:
        code_idx: The code being analyzed.
        clip_i: Index of the first clip.
        clip_j: Index of the second clip.
        n_matched_frames: Number of qpos-matched frame pairs.
        successor_dist_i: Successor distribution from clip i's matched frames.
        successor_dist_j: Successor distribution from clip j's matched frames.
        avg_qpos_distance: Mean L2 distance of matched qpos pairs.
        matched_pairs: List of matched frame pairs for video rendering.
    """

    code_idx: int
    clip_i: int
    clip_j: int
    n_matched_frames: int
    successor_dist_i: np.ndarray
    successor_dist_j: np.ndarray
    avg_qpos_distance: float
    matched_pairs: list[MatchedFramePair]


def compute_code_popularity(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> dict[int, int]:
    """Compute total frame count for each code across all clips."""
    frame_counts: dict[int, int] = {i: 0 for i in range(num_codes)}
    for result in results:
        for code_idx in result.code_indices:
            frame_counts[int(code_idx)] += 1
    return frame_counts


def get_top_k_codes(
    frame_counts: dict[int, int],
    k: int,
) -> list[tuple[int, int]]:
    """Get top K codes by frame count."""
    sorted_codes = sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_codes[:k]


def compute_transition_context(
    result: InferenceResult,
    code_idx: int,
    num_codes: int,
) -> CodeTransitionContext | None:
    """Compute transition context for a specific code in a specific clip.

    Handles stickiness by looking at transitions between runs of codes,
    not individual frames. For each contiguous run of the target code,
    the predecessor is the last different code before the run, and the
    successor is the first different code after the run.

    Args:
        result: InferenceResult for the clip.
        code_idx: The code to analyze.
        num_codes: Total number of codes.

    Returns:
        CodeTransitionContext or None if code doesn't appear in clip.
    """
    indices = result.code_indices
    n_frames = len(indices)

    predecessor_counts = np.zeros(num_codes, dtype=np.int32)
    successor_counts = np.zeros(num_codes, dtype=np.int32)
    occurrence_count = 0

    # Find runs of the target code and count transitions between runs
    i = 0
    while i < n_frames:
        if int(indices[i]) == code_idx:
            run_start = i
            # Find end of this run (first frame with different code)
            while i < n_frames and int(indices[i]) == code_idx:
                occurrence_count += 1
                i += 1
            run_end = i

            # Predecessor: last different code before this run
            if run_start > 0:
                pred_code = int(indices[run_start - 1])
                predecessor_counts[pred_code] += 1

            # Successor: first different code after this run
            if run_end < n_frames:
                succ_code = int(indices[run_end])
                successor_counts[succ_code] += 1
        else:
            i += 1

    if occurrence_count == 0:
        return None

    # Normalize to distributions
    pred_sum = predecessor_counts.sum()
    succ_sum = successor_counts.sum()

    predecessor_dist = predecessor_counts / pred_sum if pred_sum > 0 else predecessor_counts.astype(np.float64)
    successor_dist = successor_counts / succ_sum if succ_sum > 0 else successor_counts.astype(np.float64)

    return CodeTransitionContext(
        code_idx=code_idx,
        clip_idx=result.clip_idx,
        predecessor_dist=predecessor_dist,
        successor_dist=successor_dist,
        occurrence_count=occurrence_count,
        predecessor_counts=predecessor_counts,
        successor_counts=successor_counts,
    )


def compute_context_similarity(
    ctx1: CodeTransitionContext,
    ctx2: CodeTransitionContext,
) -> tuple[float, float]:
    """Compute cosine similarity between two contexts.

    Returns:
        Tuple of (predecessor_similarity, successor_similarity).
    """
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    pred_sim = cosine_sim(ctx1.predecessor_dist, ctx2.predecessor_dist)
    succ_sim = cosine_sim(ctx1.successor_dist, ctx2.successor_dist)

    return pred_sim, succ_sim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_conditional_transition_context(
    result_i: InferenceResult,
    result_j: InferenceResult,
    code_idx: int,
    num_codes: int,
    qpos_threshold: float,
) -> ConditionalTransitionContext | None:
    """Compare transitions between two clips, conditioned on similar qpos.

    For each frame in clip_i where code_idx is active, find the closest qpos
    match in clip_j (where the same code is active). Only keep matches where
    the mean absolute difference in joint angles is below threshold.

    Args:
        result_i: InferenceResult for first clip.
        result_j: InferenceResult for second clip.
        code_idx: The code to analyze.
        num_codes: Total number of codes.
        qpos_threshold: Mean absolute difference threshold for qpos matching.

    Returns:
        ConditionalTransitionContext or None if insufficient matches.
    """
    # Get frames where code is active in each clip (with joint qpos, excluding root 7 dims)
    frames_i = [
        (t, result_i.qpos[t, 7:])
        for t, c in enumerate(result_i.code_indices)
        if int(c) == code_idx and t < len(result_i.qpos)
    ]
    frames_j = [
        (t, result_j.qpos[t, 7:])
        for t, c in enumerate(result_j.code_indices)
        if int(c) == code_idx and t < len(result_j.qpos)
    ]

    if not frames_i or not frames_j:
        return None

    # Build qpos matrix for clip_j for efficient distance computation
    qpos_j_matrix = np.array([qpos for _, qpos in frames_j])  # [N_j, n_joints]
    times_j = [t for t, _ in frames_j]

    matched_pairs = []

    for t_i, qpos_i in frames_i:
        # Compute L2 distance to all frames in clip_j
        diffs = qpos_j_matrix - qpos_i  # [N_j, n_joints]
        l2_distances = np.linalg.norm(diffs, axis=1)  # [N_j]

        # Find closest match
        best_idx = np.argmin(l2_distances)

        # Check threshold (convert to mean absolute difference)
        mean_abs_diff = np.mean(np.abs(diffs[best_idx]))

        if mean_abs_diff < qpos_threshold:
            t_j = times_j[best_idx]

            # Get successor codes (if exist)
            succ_i = (
                int(result_i.code_indices[t_i + 1])
                if t_i + 1 < len(result_i.code_indices)
                else -1
            )
            succ_j = (
                int(result_j.code_indices[t_j + 1])
                if t_j + 1 < len(result_j.code_indices)
                else -1
            )

            if succ_i >= 0 and succ_j >= 0:
                matched_pairs.append(MatchedFramePair(
                    frame_i=t_i,
                    frame_j=t_j,
                    qpos_distance=mean_abs_diff,
                    succ_i=succ_i,
                    succ_j=succ_j,
                ))

    if len(matched_pairs) < 2:  # Need minimum matches
        return None

    # Compute successor distributions
    dist_i = np.zeros(num_codes)
    dist_j = np.zeros(num_codes)
    for mp in matched_pairs:
        dist_i[mp.succ_i] += 1
        dist_j[mp.succ_j] += 1

    dist_i = dist_i / dist_i.sum() if dist_i.sum() > 0 else dist_i
    dist_j = dist_j / dist_j.sum() if dist_j.sum() > 0 else dist_j

    return ConditionalTransitionContext(
        code_idx=code_idx,
        clip_i=result_i.clip_idx,
        clip_j=result_j.clip_idx,
        n_matched_frames=len(matched_pairs),
        successor_dist_i=dist_i,
        successor_dist_j=dist_j,
        avg_qpos_distance=float(np.mean([mp.qpos_distance for mp in matched_pairs])),
        matched_pairs=matched_pairs,
    )


def compute_conditional_similarity_for_code(
    results: Sequence[InferenceResult],
    code_idx: int,
    num_codes: int,
    qpos_threshold: float,
) -> dict | None:
    """Compute pairwise conditional similarities for a code across all clips.

    Args:
        results: List of InferenceResult from rollouts.
        code_idx: The code to analyze.
        num_codes: Total number of codes.
        qpos_threshold: Mean absolute difference threshold for qpos matching.

    Returns:
        Dictionary with similarity statistics or None if no valid pairs.
    """
    contexts = []
    for i, result_i in enumerate(results):
        for j, result_j in enumerate(results):
            if i >= j:  # Only upper triangle
                continue
            ctx = compute_conditional_transition_context(
                result_i, result_j, code_idx, num_codes, qpos_threshold
            )
            if ctx is not None:
                contexts.append(ctx)

    if not contexts:
        return None

    # Compute average similarity
    similarities = []
    for ctx in contexts:
        sim = cosine_similarity(ctx.successor_dist_i, ctx.successor_dist_j)
        similarities.append(sim)

    return {
        "code_idx": code_idx,
        "n_pairs": len(contexts),
        "total_matched_frames": sum(ctx.n_matched_frames for ctx in contexts),
        "avg_conditional_similarity": float(np.mean(similarities)),
        "std_conditional_similarity": float(np.std(similarities)),
        "avg_qpos_distance": float(np.mean([ctx.avg_qpos_distance for ctx in contexts])),
        "contexts": contexts,
    }


def plot_conditional_context_comparison(
    code_idx: int,
    conditional_data: dict,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Create visualization for conditional transition analysis.

    Shows:
    - Left: Heatmap of conditional similarity matrix (clip pairs)
    - Center: Bar chart of matched frame counts per clip pair
    - Right: Summary statistics box

    Args:
        code_idx: The code being analyzed.
        conditional_data: Dictionary from compute_conditional_similarity_for_code.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    contexts = conditional_data["contexts"]
    n_contexts = len(contexts)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # === Left: Similarity matrix (pairwise) ===
    ax = axes[0]

    # Build similarity matrix from contexts
    clip_indices = set()
    for ctx in contexts:
        clip_indices.add(ctx.clip_i)
        clip_indices.add(ctx.clip_j)
    clip_indices = sorted(clip_indices)
    n_clips = len(clip_indices)
    clip_to_idx = {c: i for i, c in enumerate(clip_indices)}

    sim_matrix = np.full((n_clips, n_clips), np.nan)
    for ctx in contexts:
        i = clip_to_idx[ctx.clip_i]
        j = clip_to_idx[ctx.clip_j]
        sim = cosine_similarity(ctx.successor_dist_i, ctx.successor_dist_j)
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim  # Symmetric

    # Fill diagonal with 1.0
    np.fill_diagonal(sim_matrix, 1.0)

    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_clips))
    ax.set_yticks(range(n_clips))
    ax.set_xticklabels([f"C{c}" for c in clip_indices], fontsize=7)
    ax.set_yticklabels([f"C{c}" for c in clip_indices], fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Cosine Sim")
    ax.set_title("Conditional Similarity\n(qpos-matched)", fontsize=10, fontweight="bold")

    # === Center: Bar chart of matched frames ===
    ax = axes[1]

    pair_labels = [f"C{ctx.clip_i}-C{ctx.clip_j}" for ctx in contexts]
    matched_counts = [ctx.n_matched_frames for ctx in contexts]
    pair_sims = [cosine_similarity(ctx.successor_dist_i, ctx.successor_dist_j) for ctx in contexts]

    # Color bars by similarity
    colors = plt.cm.RdYlGn([s for s in pair_sims])

    bars = ax.bar(range(n_contexts), matched_counts, color=colors, edgecolor="white")
    ax.set_xticks(range(n_contexts))
    ax.set_xticklabels(pair_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Matched Frames", fontsize=9)
    ax.set_title("Matched Frame Counts\n(per clip pair)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # === Right: Summary statistics ===
    ax = axes[2]
    ax.axis("off")

    avg_sim = conditional_data["avg_conditional_similarity"]
    std_sim = conditional_data["std_conditional_similarity"]
    total_matched = conditional_data["total_matched_frames"]
    avg_qpos_dist = conditional_data["avg_qpos_distance"]
    n_pairs = conditional_data["n_pairs"]

    # Determine interpretation
    if avg_sim > 0.8:
        interpretation = "STRUCTURED\nSame pose → consistent transitions"
    elif avg_sim > 0.5:
        interpretation = "PARTIAL\nSome pose-dependent consistency"
    else:
        interpretation = "UNSTRUCTURED\nSimilar poses → different transitions"

    stats_text = f"""Code Index: {code_idx}

Clip Pairs Analyzed: {n_pairs}
Total Matched Frames: {total_matched}

Avg Qpos Distance: {avg_qpos_dist:.4f}

Conditional Similarity:
  Mean: {avg_sim:.3f}
  Std:  {std_sim:.3f}

Interpretation:
{interpretation}
"""

    ax.text(
        0.1, 0.9, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"Conditional Transition Analysis: Code {code_idx}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    return fig


def extract_transition_segments(
    result: InferenceResult,
    code_idx: int,
    context_frames: int = 10,
    max_segments: int = 3,
) -> list[TransitionSegment]:
    """Extract segments showing transitions through a specific code.

    Args:
        result: InferenceResult for the clip.
        code_idx: The code to find transitions for.
        context_frames: Number of frames before/after to include.
        max_segments: Maximum segments to extract per clip.

    Returns:
        List of TransitionSegment objects.
    """
    indices = result.code_indices
    n_frames = len(indices)

    segments = []

    # Find all occurrences of this code
    i = 0
    while i < n_frames and len(segments) < max_segments:
        if int(indices[i]) == code_idx:
            # Find the run of this code
            run_start = i
            while i < n_frames and int(indices[i]) == code_idx:
                i += 1
            run_end = i

            # Get predecessor and successor codes
            pred_code = int(indices[run_start - 1]) if run_start > 0 else -1
            succ_code = int(indices[run_end]) if run_end < n_frames else -1

            # Skip if no clear predecessor/successor
            if pred_code == -1 or succ_code == -1:
                continue

            # Expand to include context
            seg_start = max(0, run_start - context_frames)
            seg_end = min(n_frames, run_end + context_frames)

            segments.append(TransitionSegment(
                code_idx=code_idx,
                clip_idx=result.clip_idx,
                start_frame=seg_start,
                end_frame=seg_end,
                predecessor_code=pred_code,
                successor_code=succ_code,
                code_indices=indices[seg_start:seg_end].copy(),
            ))
        else:
            i += 1

    return segments


def render_transition_video(
    env: Any,
    result: InferenceResult,
    segment: TransitionSegment,
    output_path: Path,
    num_codes: int,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    bar_height: int = 40,
) -> str:
    """Render a transition segment video with code timeline bar.

    Args:
        env: Environment with mj_model attribute.
        result: InferenceResult with qpos data.
        segment: TransitionSegment to render.
        output_path: Path to save video.
        num_codes: Total number of codes.
        camera: Camera name.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        bar_height: Height of the code timeline bar.

    Returns:
        Path to rendered video.
    """
    import imageio
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if result.qpos is None or len(result.qpos) == 0:
        logging.warning(f"No qpos data for clip {segment.clip_idx}")
        return ""

    # Validate segment bounds
    if segment.start_frame >= len(result.qpos) or segment.end_frame > len(result.qpos):
        logging.warning(f"Segment bounds out of range for clip {segment.clip_idx}")
        return ""

    code_colors = get_nature_colormap(num_codes)

    # Setup MuJoCo renderer
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    render_height = height - bar_height
    renderer = mujoco.Renderer(mj_model, height=render_height, width=width)

    # Get camera ID
    cam_id = -1
    if camera:
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        except Exception:
            logging.warning(f"Camera '{camera}' not found, using default")

    n_frames = segment.end_frame - segment.start_frame
    indices = segment.code_indices

    if n_frames == 0 or len(indices) == 0:
        logging.warning(f"Empty segment for clip {segment.clip_idx}")
        renderer.close()
        return ""

    frames = []
    for i in range(n_frames):
        frame_idx = segment.start_frame + i

        # Set qpos and forward kinematics
        mj_data.qpos[:] = result.qpos[frame_idx]
        mujoco.mj_forward(mj_model, mj_data)

        # Render frame
        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)
        render_frame = renderer.render()

        # Create full frame with bar
        full_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        full_frame[:render_height, :] = render_frame

        # Draw code timeline bar
        bar_y = render_height
        for j in range(n_frames):
            x_start = int(j * width / n_frames)
            x_end = int((j + 1) * width / n_frames)
            code_idx = int(indices[j]) if j < len(indices) else 0
            color = code_colors[code_idx]

            # Highlight the target code with a white border
            if code_idx == segment.code_idx:
                full_frame[bar_y:bar_y + 2, x_start:x_end] = [255, 255, 255]
                full_frame[bar_y + bar_height - 2:bar_y + bar_height, x_start:x_end] = [255, 255, 255]
                full_frame[bar_y + 2:bar_y + bar_height - 2, x_start:x_end] = color
            else:
                full_frame[bar_y:bar_y + bar_height, x_start:x_end] = color

        # Playhead
        playhead_x = int(i * width / n_frames)
        full_frame[bar_y:bar_y + bar_height, playhead_x:playhead_x + 2] = [255, 255, 255]

        frames.append(full_frame)

    renderer.close()

    if len(frames) == 0:
        return ""

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return str(output_path)


def render_conditional_comparison_video(
    env: Any,
    result_i: InferenceResult,
    result_j: InferenceResult,
    matched_pair: MatchedFramePair,
    code_idx: int,
    output_path: Path,
    num_codes: int,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    context_frames: int = 15,
    bar_height: int = 40,
) -> str:
    """Render side-by-side comparison video of matched frames from two clips.

    Shows the transition around matched frames where qpos is similar.

    Args:
        env: Environment with mj_model attribute.
        result_i: InferenceResult for first clip.
        result_j: InferenceResult for second clip.
        matched_pair: MatchedFramePair with frame indices.
        code_idx: The code being analyzed.
        output_path: Path to save video.
        num_codes: Total number of codes.
        camera: Camera name.
        width: Video width (for each side, total will be 2x).
        height: Video height.
        fps: Frames per second.
        context_frames: Number of frames before/after to include.
        bar_height: Height of the code timeline bar.

    Returns:
        Path to rendered video.
    """
    import imageio
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate data
    if result_i.qpos is None or result_j.qpos is None:
        return ""

    code_colors = get_nature_colormap(num_codes)

    # Setup MuJoCo renderer
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    half_width = width // 2
    render_height = height - bar_height
    renderer = mujoco.Renderer(mj_model, height=render_height, width=half_width)

    # Get camera ID
    cam_id = -1
    if camera:
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        except Exception:
            pass

    # Compute frame ranges for both clips
    start_i = max(0, matched_pair.frame_i - context_frames)
    end_i = min(len(result_i.qpos), matched_pair.frame_i + context_frames + 1)
    start_j = max(0, matched_pair.frame_j - context_frames)
    end_j = min(len(result_j.qpos), matched_pair.frame_j + context_frames + 1)

    n_frames = max(end_i - start_i, end_j - start_j)

    if n_frames == 0:
        renderer.close()
        return ""

    frames = []
    for i in range(n_frames):
        # Create combined frame (side by side)
        full_frame = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark gray background

        # Render clip i (left side)
        frame_idx_i = start_i + min(i, end_i - start_i - 1)
        mj_data.qpos[:] = result_i.qpos[frame_idx_i]
        mujoco.mj_forward(mj_model, mj_data)
        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)
        render_i = renderer.render()
        full_frame[:render_height, :half_width] = render_i

        # Render clip j (right side)
        frame_idx_j = start_j + min(i, end_j - start_j - 1)
        mj_data.qpos[:] = result_j.qpos[frame_idx_j]
        mujoco.mj_forward(mj_model, mj_data)
        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)
        render_j = renderer.render()
        full_frame[:render_height, half_width:] = render_j

        # Draw divider line
        full_frame[:render_height, half_width - 1:half_width + 1] = [100, 100, 100]

        # Draw code timeline bars
        bar_y = render_height

        # Left bar (clip i)
        for j in range(end_i - start_i):
            x_start = int(j * half_width / (end_i - start_i))
            x_end = int((j + 1) * half_width / (end_i - start_i))
            idx = start_i + j
            if idx < len(result_i.code_indices):
                c_idx = int(result_i.code_indices[idx])
                color = code_colors[c_idx]
                if c_idx == code_idx:
                    full_frame[bar_y:bar_y + 2, x_start:x_end] = [255, 255, 255]
                    full_frame[bar_y + bar_height - 2:bar_y + bar_height, x_start:x_end] = [255, 255, 255]
                    full_frame[bar_y + 2:bar_y + bar_height - 2, x_start:x_end] = color
                else:
                    full_frame[bar_y:bar_y + bar_height, x_start:x_end] = color

        # Right bar (clip j)
        for j in range(end_j - start_j):
            x_start = half_width + int(j * half_width / (end_j - start_j))
            x_end = half_width + int((j + 1) * half_width / (end_j - start_j))
            idx = start_j + j
            if idx < len(result_j.code_indices):
                c_idx = int(result_j.code_indices[idx])
                color = code_colors[c_idx]
                if c_idx == code_idx:
                    full_frame[bar_y:bar_y + 2, x_start:x_end] = [255, 255, 255]
                    full_frame[bar_y + bar_height - 2:bar_y + bar_height, x_start:x_end] = [255, 255, 255]
                    full_frame[bar_y + 2:bar_y + bar_height - 2, x_start:x_end] = color
                else:
                    full_frame[bar_y:bar_y + bar_height, x_start:x_end] = color

        # Playheads
        if end_i > start_i:
            playhead_i = int(min(i, end_i - start_i - 1) * half_width / (end_i - start_i))
            full_frame[bar_y:bar_y + bar_height, playhead_i:playhead_i + 2] = [255, 255, 255]
        if end_j > start_j:
            playhead_j = half_width + int(min(i, end_j - start_j - 1) * half_width / (end_j - start_j))
            full_frame[bar_y:bar_y + bar_height, playhead_j:playhead_j + 2] = [255, 255, 255]

        frames.append(full_frame)

    renderer.close()

    if len(frames) == 0:
        return ""

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return str(output_path)


def plot_code_context_comparison(
    code_idx: int,
    contexts: list[CodeTransitionContext],
    num_codes: int,
    figsize: tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Create visualization comparing transition contexts across clips.

    Args:
        code_idx: The code being analyzed.
        contexts: List of CodeTransitionContext from different clips.
        num_codes: Total number of codes.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_clips = len(contexts)
    code_colors = get_nature_colormap(num_codes)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # === Row 0, Col 0-1: Predecessor distributions across clips ===
    ax = fig.add_subplot(gs[0, :2])

    # Stack predecessor distributions
    pred_matrix = np.array([ctx.predecessor_dist for ctx in contexts])

    # Find non-zero columns (codes that appear as predecessors)
    nonzero_cols = np.where(pred_matrix.sum(axis=0) > 0)[0]

    if len(nonzero_cols) > 0:
        pred_subset = pred_matrix[:, nonzero_cols]
        im = ax.imshow(pred_subset, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(nonzero_cols)))
        ax.set_xticklabels(nonzero_cols, fontsize=7, rotation=45)
        ax.set_yticks(range(n_clips))
        ax.set_yticklabels([f"Clip {ctx.clip_idx}" for ctx in contexts], fontsize=8)
        ax.set_xlabel("Predecessor Code", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.6, label="P(pred)")

    ax.set_title(f"Code {code_idx}: Predecessor Distributions Across Clips", fontsize=10, fontweight="bold")

    # === Row 0, Col 2: Predecessor similarity matrix ===
    ax = fig.add_subplot(gs[0, 2])

    pred_sim_matrix = np.zeros((n_clips, n_clips))
    for i in range(n_clips):
        for j in range(n_clips):
            pred_sim, _ = compute_context_similarity(contexts[i], contexts[j])
            pred_sim_matrix[i, j] = pred_sim

    im = ax.imshow(pred_sim_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_clips))
    ax.set_yticks(range(n_clips))
    ax.set_xticklabels([f"C{ctx.clip_idx}" for ctx in contexts], fontsize=7)
    ax.set_yticklabels([f"C{ctx.clip_idx}" for ctx in contexts], fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Cosine Sim")
    ax.set_title("Predecessor Similarity", fontsize=10, fontweight="bold")

    # === Row 1, Col 0-1: Successor distributions across clips ===
    ax = fig.add_subplot(gs[1, :2])

    succ_matrix = np.array([ctx.successor_dist for ctx in contexts])
    nonzero_cols = np.where(succ_matrix.sum(axis=0) > 0)[0]

    if len(nonzero_cols) > 0:
        succ_subset = succ_matrix[:, nonzero_cols]
        im = ax.imshow(succ_subset, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
        ax.set_xticks(range(len(nonzero_cols)))
        ax.set_xticklabels(nonzero_cols, fontsize=7, rotation=45)
        ax.set_yticks(range(n_clips))
        ax.set_yticklabels([f"Clip {ctx.clip_idx}" for ctx in contexts], fontsize=8)
        ax.set_xlabel("Successor Code", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.6, label="P(succ)")

    ax.set_title(f"Code {code_idx}: Successor Distributions Across Clips", fontsize=10, fontweight="bold")

    # === Row 1, Col 2: Successor similarity matrix ===
    ax = fig.add_subplot(gs[1, 2])

    succ_sim_matrix = np.zeros((n_clips, n_clips))
    for i in range(n_clips):
        for j in range(n_clips):
            _, succ_sim = compute_context_similarity(contexts[i], contexts[j])
            succ_sim_matrix[i, j] = succ_sim

    im = ax.imshow(succ_sim_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_clips))
    ax.set_yticks(range(n_clips))
    ax.set_xticklabels([f"C{ctx.clip_idx}" for ctx in contexts], fontsize=7)
    ax.set_yticklabels([f"C{ctx.clip_idx}" for ctx in contexts], fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Cosine Sim")
    ax.set_title("Successor Similarity", fontsize=10, fontweight="bold")

    # === Row 2, Col 0: Combined similarity (average of pred + succ) ===
    ax = fig.add_subplot(gs[2, 0])

    combined_sim = (pred_sim_matrix + succ_sim_matrix) / 2
    im = ax.imshow(combined_sim, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_clips))
    ax.set_yticks(range(n_clips))
    ax.set_xticklabels([f"C{ctx.clip_idx}" for ctx in contexts], fontsize=7)
    ax.set_yticklabels([f"C{ctx.clip_idx}" for ctx in contexts], fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Avg Sim")
    ax.set_title("Combined Context Similarity", fontsize=10, fontweight="bold")

    # === Row 2, Col 1: Occurrence counts per clip ===
    ax = fig.add_subplot(gs[2, 1])

    clip_labels = [f"Clip {ctx.clip_idx}" for ctx in contexts]
    counts = [ctx.occurrence_count for ctx in contexts]
    colors = [code_colors[code_idx] / 255.0] * n_clips

    ax.bar(range(n_clips), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(n_clips))
    ax.set_xticklabels(clip_labels, fontsize=7, rotation=45)
    ax.set_ylabel("Frame Count", fontsize=9)
    ax.set_title(f"Code {code_idx} Usage Per Clip", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # === Row 2, Col 2: Summary statistics ===
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    # Compute summary stats
    avg_pred_sim = pred_sim_matrix[np.triu_indices(n_clips, k=1)].mean() if n_clips > 1 else 1.0
    avg_succ_sim = succ_sim_matrix[np.triu_indices(n_clips, k=1)].mean() if n_clips > 1 else 1.0
    avg_combined = (avg_pred_sim + avg_succ_sim) / 2

    total_occurrences = sum(ctx.occurrence_count for ctx in contexts)
    clips_present = n_clips

    # Determine consistency label
    if avg_combined > 0.8:
        consistency = "HIGH - Same function"
    elif avg_combined > 0.5:
        consistency = "MEDIUM - Partial overlap"
    else:
        consistency = "LOW - Context-dependent"

    stats_text = f"""Code Index: {code_idx}

Clips Present: {clips_present}
Total Frames: {total_occurrences}

Avg Predecessor Sim: {avg_pred_sim:.3f}
Avg Successor Sim: {avg_succ_sim:.3f}
Avg Combined Sim: {avg_combined:.3f}

Consistency: {consistency}
"""

    ax.text(0.1, 0.9, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"Transition Context Analysis: Code {code_idx}",
                 fontsize=14, fontweight="bold", y=0.98)

    return fig


def figure_to_base64(fig: plt.Figure, fmt: str = "png", dpi: int = 100) -> str:
    """Convert matplotlib figure to base64-encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64


def video_to_base64(video_path: str | Path) -> str:
    """Convert video file to base64-encoded data URL."""
    video_path = Path(video_path)
    if not video_path.exists():
        return ""
    with open(video_path, "rb") as f:
        video_data = f.read()
    b64 = base64.b64encode(video_data).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def generate_context_html(
    code_analyses: list[dict],
    output_path: Path,
    output_dir: Path,
    title: str = "Transition Context Analysis",
) -> str:
    """Generate interactive HTML for browsing code context comparisons.

    Args:
        code_analyses: List of analysis data per code.
        output_path: Path to save HTML file.
        output_dir: Base output directory (to resolve video paths).
        title: HTML page title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert video paths to base64 data URLs for browser compatibility
    for ca in code_analyses:
        for video in ca.get("videos", []):
            rel_path = video.get("path", "")
            if rel_path:
                full_path = output_dir / rel_path
                video["data_url"] = video_to_base64(full_path)

    js_data = json.dumps(code_analyses)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #fff;
            font-size: 26px;
        }}
        .controls {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .slider-row {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .nav-btn {{
            background: rgba(255, 152, 0, 0.3);
            border: 1px solid rgba(255, 152, 0, 0.5);
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .nav-btn:hover {{ background: rgba(255, 152, 0, 0.5); }}
        .nav-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
        .code-counter {{
            font-size: 18px;
            font-weight: 600;
            color: #ff9800;
            min-width: 120px;
            text-align: center;
        }}
        .slider-wrapper {{ flex: 1; min-width: 200px; }}
        input[type="range"] {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #3a3a5a;
            outline: none;
            -webkit-appearance: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #ff9800;
            cursor: pointer;
        }}
        .stats-row {{
            display: flex;
            gap: 12px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: rgba(255, 152, 0, 0.15);
            border: 1px solid rgba(255, 152, 0, 0.3);
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 13px;
        }}
        .stat-badge strong {{ color: #ff9800; }}
        .stat-badge.high {{ background: rgba(76, 175, 80, 0.2); border-color: rgba(76, 175, 80, 0.4); }}
        .stat-badge.medium {{ background: rgba(255, 193, 7, 0.2); border-color: rgba(255, 193, 7, 0.4); }}
        .stat-badge.low {{ background: rgba(244, 67, 54, 0.2); border-color: rgba(244, 67, 54, 0.4); }}
        .content-area {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .image-container {{
            flex: 2;
            min-width: 600px;
            background: #fff;
            border-radius: 10px;
            padding: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 6px;
        }}
        .videos-container {{
            flex: 1;
            min-width: 300px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
        }}
        .videos-container h3 {{
            color: #ff9800;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        .video-item {{
            margin-bottom: 15px;
        }}
        .video-item video {{
            width: 100%;
            border-radius: 6px;
        }}
        .video-label {{
            font-size: 11px;
            color: #aaa;
            margin-top: 5px;
        }}
        .hint {{
            font-size: 11px;
            color: #666;
            margin-top: 10px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="controls">
            <div class="slider-row">
                <button class="nav-btn" id="prevBtn" onclick="prevCode()">&#9664; Prev</button>
                <span class="code-counter" id="codeCounter">Code 0</span>
                <button class="nav-btn" id="nextBtn" onclick="nextCode()">Next &#9654;</button>
                <div class="slider-wrapper">
                    <input type="range" id="codeSlider" min="0" max="{len(code_analyses) - 1}" value="0" oninput="updateCode(this.value)">
                </div>
            </div>
            <div class="stats-row" id="statsRow"></div>
            <div class="hint">Use &#8592; &#8594; arrow keys or slider to navigate between codes</div>
        </div>
        <div class="content-area">
            <div class="image-container">
                <img id="analysisImage" src="" alt="Context Analysis">
            </div>
            <div class="videos-container">
                <h3>Transition Examples</h3>
                <div id="videosArea"></div>
            </div>
        </div>
    </div>
    <script>
        const codesData = {js_data};
        let idx = 0;

        function getConsistencyClass(sim) {{
            if (sim > 0.8) return 'high';
            if (sim > 0.5) return 'medium';
            return 'low';
        }}

        function updateCode(i) {{
            idx = parseInt(i);
            const c = codesData[idx];
            document.getElementById('analysisImage').src = 'data:image/png;base64,' + c.image;
            document.getElementById('codeCounter').textContent = 'Code ' + c.code_idx + ' (' + (idx + 1) + '/' + codesData.length + ')';
            document.getElementById('codeSlider').value = idx;

            const consistencyClass = getConsistencyClass(c.stats.avg_combined_sim);
            document.getElementById('statsRow').innerHTML = `
                <div class="stat-badge"><strong>Code:</strong> ${{c.code_idx}}</div>
                <div class="stat-badge"><strong>Total Frames:</strong> ${{c.stats.total_frames}}</div>
                <div class="stat-badge"><strong>Clips:</strong> ${{c.stats.n_clips}}</div>
                <div class="stat-badge"><strong>Pred Sim:</strong> ${{c.stats.avg_pred_sim.toFixed(3)}}</div>
                <div class="stat-badge"><strong>Succ Sim:</strong> ${{c.stats.avg_succ_sim.toFixed(3)}}</div>
                <div class="stat-badge ${{consistencyClass}}"><strong>Combined:</strong> ${{c.stats.avg_combined_sim.toFixed(3)}}</div>
            `;

            // Update videos (use data_url for embedded base64 video)
            let videosHtml = '';
            if (c.videos && c.videos.length > 0) {{
                c.videos.forEach((v, vi) => {{
                    const videoSrc = v.data_url || v.path;
                    if (videoSrc) {{
                        videosHtml += `
                            <div class="video-item">
                                <video controls loop muted autoplay>
                                    <source src="${{videoSrc}}" type="video/mp4">
                                </video>
                                <div class="video-label">Clip ${{v.clip_idx}}: ${{v.pred}} &#8594; ${{c.code_idx}} &#8594; ${{v.succ}}</div>
                            </div>
                        `;
                    }}
                }});
            }}
            if (!videosHtml) {{
                videosHtml = '<div class="video-label">No transition videos available</div>';
            }}
            document.getElementById('videosArea').innerHTML = videosHtml;

            document.getElementById('prevBtn').disabled = idx === 0;
            document.getElementById('nextBtn').disabled = idx === codesData.length - 1;
        }}

        function nextCode() {{ if (idx < codesData.length - 1) updateCode(idx + 1); }}
        function prevCode() {{ if (idx > 0) updateCode(idx - 1); }}

        document.addEventListener('keydown', e => {{
            if (e.key === 'ArrowRight') nextCode();
            else if (e.key === 'ArrowLeft') prevCode();
        }});

        updateCode(0);
    </script>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(html)

    return str(output_path)


def generate_conditional_html(
    code_analyses: list[dict],
    output_path: Path,
    output_dir: Path,
    title: str = "Conditional Transition Analysis",
) -> str:
    """Generate standalone HTML for conditional (qpos-matched) transition analysis.

    Args:
        code_analyses: List of analysis data per code (must have 'conditional' key).
        output_path: Path to save HTML file.
        output_dir: Base output directory (to resolve video paths).
        title: HTML page title.

    Returns:
        Path to generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to only codes with conditional analysis
    conditional_codes = [ca for ca in code_analyses if ca.get("conditional")]

    # Convert video paths to base64 for embedding
    for ca in conditional_codes:
        cond = ca.get("conditional", {})
        for video in cond.get("videos", []):
            rel_path = video.get("path", "")
            if rel_path:
                full_path = output_dir / rel_path
                video["data_url"] = video_to_base64(full_path)

    if not conditional_codes:
        # Create a simple "no data" HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .message {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="message">
        <h2>No Conditional Analysis Data</h2>
        <p>Insufficient qpos-matched frames for conditional transition analysis.</p>
    </div>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html)
        return str(output_path)

    js_data = json.dumps(conditional_codes)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #fff;
            font-size: 26px;
        }}
        .subtitle {{
            text-align: center;
            color: #4fc3f7;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .controls {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .slider-row {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .nav-btn {{
            background: rgba(79, 195, 247, 0.3);
            border: 1px solid rgba(79, 195, 247, 0.5);
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .nav-btn:hover {{ background: rgba(79, 195, 247, 0.5); }}
        .nav-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
        .code-counter {{
            font-size: 18px;
            font-weight: 600;
            color: #4fc3f7;
            min-width: 120px;
            text-align: center;
        }}
        .slider-wrapper {{ flex: 1; min-width: 200px; }}
        input[type="range"] {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #3a3a5a;
            outline: none;
            -webkit-appearance: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #4fc3f7;
            cursor: pointer;
        }}
        .stats-row {{
            display: flex;
            gap: 12px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: rgba(79, 195, 247, 0.15);
            border: 1px solid rgba(79, 195, 247, 0.3);
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 13px;
        }}
        .stat-badge strong {{ color: #4fc3f7; }}
        .stat-badge.high {{ background: rgba(76, 175, 80, 0.2); border-color: rgba(76, 175, 80, 0.4); }}
        .stat-badge.medium {{ background: rgba(255, 193, 7, 0.2); border-color: rgba(255, 193, 7, 0.4); }}
        .stat-badge.low {{ background: rgba(244, 67, 54, 0.2); border-color: rgba(244, 67, 54, 0.4); }}
        .image-container {{
            background: #fff;
            border-radius: 10px;
            padding: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 6px;
        }}
        .hint {{
            font-size: 11px;
            color: #666;
            margin-top: 10px;
            text-align: center;
        }}
        .interpretation {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(79, 195, 247, 0.1);
            border-radius: 8px;
            border-left: 4px solid #4fc3f7;
        }}
        .interpretation h4 {{
            color: #4fc3f7;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .interpretation p {{
            font-size: 12px;
            line-height: 1.5;
            color: #aaa;
        }}
        .content-area {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 20px;
        }}
        .image-panel {{
            flex: 2;
            min-width: 600px;
        }}
        .videos-panel {{
            flex: 1;
            min-width: 300px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
        }}
        .videos-panel h3 {{
            color: #4fc3f7;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        .video-item {{
            margin-bottom: 15px;
        }}
        .video-item video {{
            width: 100%;
            border-radius: 6px;
        }}
        .video-label {{
            font-size: 11px;
            color: #aaa;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">Comparing code transitions when clips have similar joint positions (qpos[7:])</p>
        <div class="controls">
            <div class="slider-row">
                <button class="nav-btn" id="prevBtn" onclick="prevCode()">&#9664; Prev</button>
                <span class="code-counter" id="codeCounter">Code 0</span>
                <button class="nav-btn" id="nextBtn" onclick="nextCode()">Next &#9654;</button>
                <div class="slider-wrapper">
                    <input type="range" id="codeSlider" min="0" max="{len(conditional_codes) - 1}" value="0" oninput="updateCode(this.value)">
                </div>
            </div>
            <div class="stats-row" id="statsRow"></div>
            <div class="hint">Use &#8592; &#8594; arrow keys or slider to navigate between codes</div>
        </div>
        <div class="content-area">
            <div class="image-panel">
                <div class="image-container">
                    <img id="analysisImage" src="" alt="Conditional Analysis">
                </div>
            </div>
            <div class="videos-panel">
                <h3>Matched Pose Comparisons</h3>
                <div id="videosArea"></div>
            </div>
        </div>
        <div class="interpretation">
            <h4>How to Interpret</h4>
            <p>
                <strong>High conditional similarity (&gt;0.8):</strong> Structured superposition - same pose context leads to consistent transitions.<br>
                <strong>Medium similarity (0.5-0.8):</strong> Partial structure - some pose-dependent consistency.<br>
                <strong>Low similarity (&lt;0.5):</strong> Unstructured - even similar poses produce different transitions (different behaviors).
            </p>
        </div>
    </div>
    <script>
        const codesData = {js_data};
        let idx = 0;

        function getConsistencyClass(sim) {{
            if (sim > 0.8) return 'high';
            if (sim > 0.5) return 'medium';
            return 'low';
        }}

        function updateCode(i) {{
            idx = parseInt(i);
            const c = codesData[idx];
            const cond = c.conditional;

            document.getElementById('analysisImage').src = 'data:image/png;base64,' + cond.image;
            document.getElementById('codeCounter').textContent = 'Code ' + c.code_idx + ' (' + (idx + 1) + '/' + codesData.length + ')';
            document.getElementById('codeSlider').value = idx;

            const condClass = getConsistencyClass(cond.avg_sim);
            document.getElementById('statsRow').innerHTML = `
                <div class="stat-badge"><strong>Code:</strong> ${{c.code_idx}}</div>
                <div class="stat-badge ${{condClass}}"><strong>Conditional Sim:</strong> ${{cond.avg_sim.toFixed(3)}}</div>
                <div class="stat-badge"><strong>Std:</strong> ${{cond.std_sim.toFixed(3)}}</div>
                <div class="stat-badge"><strong>Matched Frames:</strong> ${{cond.total_matched}}</div>
                <div class="stat-badge"><strong>Clip Pairs:</strong> ${{cond.n_pairs}}</div>
                <div class="stat-badge"><strong>Avg Qpos Dist:</strong> ${{cond.avg_qpos_distance.toFixed(4)}}</div>
            `;

            // Update videos
            let videosHtml = '';
            if (cond.videos && cond.videos.length > 0) {{
                cond.videos.forEach((v, vi) => {{
                    const videoSrc = v.data_url || v.path;
                    if (videoSrc) {{
                        const succMatch = v.succ_i === v.succ_j ? '&#10003;' : '&#10007;';
                        const succClass = v.succ_i === v.succ_j ? 'high' : 'low';
                        videosHtml += `
                            <div class="video-item">
                                <video controls loop muted autoplay>
                                    <source src="${{videoSrc}}" type="video/mp4">
                                </video>
                                <div class="video-label">
                                    Clips ${{v.clip_i}} vs ${{v.clip_j}} |
                                    Succ: ${{v.succ_i}} vs ${{v.succ_j}}
                                    <span class="stat-badge ${{succClass}}" style="padding:2px 6px;font-size:10px;">${{succMatch}}</span>
                                </div>
                            </div>
                        `;
                    }}
                }});
            }}
            if (!videosHtml) {{
                videosHtml = '<div class="video-label">No comparison videos available</div>';
            }}
            document.getElementById('videosArea').innerHTML = videosHtml;

            document.getElementById('prevBtn').disabled = idx === 0;
            document.getElementById('nextBtn').disabled = idx === codesData.length - 1;
        }}

        function nextCode() {{ if (idx < codesData.length - 1) updateCode(idx + 1); }}
        function prevCode() {{ if (idx > 0) updateCode(idx - 1); }}

        document.addEventListener('keydown', e => {{
            if (e.key === 'ArrowRight') nextCode();
            else if (e.key === 'ArrowLeft') prevCode();
        }});

        updateCode(0);
    </script>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(html)

    return str(output_path)


def run_transition_context_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    top_k: int = 10,
    min_clips_for_comparison: int = 3,
    render_videos: bool = True,
    env: Any = None,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    max_videos_per_code: int = 4,
    conditional_cfg: dict | None = None,
) -> dict[str, Any]:
    """Run transition context analysis on top K most used codes.

    Args:
        results: List of InferenceResult from rollouts.
        num_codes: Total number of codes.
        output_dir: Directory to save outputs.
        top_k: Number of top codes to analyze.
        min_clips_for_comparison: Minimum clips a code must appear in.
        render_videos: Whether to render transition videos.
        env: Environment for video rendering.
        camera: Camera name.
        width: Video width.
        height: Video height.
        fps: Video FPS.
        max_videos_per_code: Maximum transition videos per code.
        conditional_cfg: Configuration for qpos-conditioned transition analysis.
            Keys: enabled (bool), qpos_threshold (float).

    Returns:
        Dictionary with html_path and analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Running transition context analysis...")

    # Get top K codes
    frame_counts = compute_code_popularity(results, num_codes)
    top_codes = get_top_k_codes(frame_counts, top_k)

    logging.info(f"  Top {len(top_codes)} codes by usage:")
    for code_idx, count in top_codes:
        logging.info(f"    Code {code_idx}: {count} frames")

    code_analyses = []

    for code_idx, total_count in top_codes:
        logging.info(f"\n  Analyzing code {code_idx}...")

        # Compute context for each clip where this code appears
        contexts = []
        for result in results:
            ctx = compute_transition_context(result, code_idx, num_codes)
            if ctx is not None and ctx.occurrence_count >= 2:  # Need at least 2 occurrences for meaningful context
                contexts.append(ctx)

        if len(contexts) < min_clips_for_comparison:
            logging.info(f"    Code {code_idx} appears in only {len(contexts)} clips, skipping")
            continue

        logging.info(f"    Found in {len(contexts)} clips")

        # Generate comparison plot
        fig = plot_code_context_comparison(code_idx, contexts, num_codes)
        img_b64 = figure_to_base64(fig, dpi=100)
        plt.close(fig)

        # Compute summary stats
        n_clips = len(contexts)
        pred_sim_sum = 0
        succ_sim_sum = 0
        n_pairs = 0

        for i in range(n_clips):
            for j in range(i + 1, n_clips):
                pred_sim, succ_sim = compute_context_similarity(contexts[i], contexts[j])
                pred_sim_sum += pred_sim
                succ_sim_sum += succ_sim
                n_pairs += 1

        avg_pred_sim = pred_sim_sum / n_pairs if n_pairs > 0 else 1.0
        avg_succ_sim = succ_sim_sum / n_pairs if n_pairs > 0 else 1.0
        avg_combined_sim = (avg_pred_sim + avg_succ_sim) / 2

        # Render transition videos
        video_info = []
        if render_videos and env is not None:
            video_dir = output_dir / "videos" / f"code_{code_idx:03d}"
            video_dir.mkdir(parents=True, exist_ok=True)

            videos_rendered = 0
            for result in results:
                if videos_rendered >= max_videos_per_code:
                    break

                segments = extract_transition_segments(result, code_idx, context_frames=15, max_segments=1)

                for seg in segments:
                    if videos_rendered >= max_videos_per_code:
                        break

                    video_path = video_dir / f"clip_{seg.clip_idx:03d}_transition.mp4"

                    try:
                        path = render_transition_video(
                            env=env,
                            result=result,
                            segment=seg,
                            output_path=video_path,
                            num_codes=num_codes,
                            camera=camera,
                            width=width,
                            height=height,
                            fps=fps,
                        )
                        if path:
                            # Use relative path for HTML
                            rel_path = f"videos/code_{code_idx:03d}/clip_{seg.clip_idx:03d}_transition.mp4"
                            video_info.append({
                                "path": rel_path,
                                "clip_idx": int(seg.clip_idx),
                                "pred": int(seg.predecessor_code),
                                "succ": int(seg.successor_code),
                            })
                            videos_rendered += 1
                    except Exception as e:
                        logging.warning(f"    Failed to render video for clip {seg.clip_idx}: {e}")

            logging.info(f"    Rendered {len(video_info)} transition videos")

        # Conditional analysis (qpos-matched transitions)
        conditional_result = None
        if conditional_cfg and conditional_cfg.get("enabled", False):
            qpos_threshold = conditional_cfg.get("qpos_threshold", 0.1)
            conditional_data = compute_conditional_similarity_for_code(
                results, code_idx, num_codes, qpos_threshold
            )
            if conditional_data:
                # Generate conditional visualization
                cond_fig = plot_conditional_context_comparison(code_idx, conditional_data)
                cond_img_b64 = figure_to_base64(cond_fig, dpi=100)
                plt.close(cond_fig)

                # Render conditional comparison videos
                cond_video_info = []
                if render_videos and env is not None:
                    cond_video_dir = output_dir / "videos" / f"code_{code_idx:03d}_conditional"
                    cond_video_dir.mkdir(parents=True, exist_ok=True)

                    # Build lookup for results by clip_idx
                    results_by_clip = {r.clip_idx: r for r in results}

                    cond_videos_rendered = 0
                    max_cond_videos = conditional_cfg.get("max_videos", 4)

                    for ctx in conditional_data["contexts"]:
                        if cond_videos_rendered >= max_cond_videos:
                            break

                        # Get results for this clip pair
                        r_i = results_by_clip.get(ctx.clip_i)
                        r_j = results_by_clip.get(ctx.clip_j)
                        if r_i is None or r_j is None:
                            continue

                        # Pick the best matched pair (lowest qpos distance)
                        if not ctx.matched_pairs:
                            continue
                        best_pair = min(ctx.matched_pairs, key=lambda p: p.qpos_distance)

                        video_path = cond_video_dir / f"clips_{ctx.clip_i:03d}_{ctx.clip_j:03d}.mp4"

                        try:
                            path = render_conditional_comparison_video(
                                env=env,
                                result_i=r_i,
                                result_j=r_j,
                                matched_pair=best_pair,
                                code_idx=code_idx,
                                output_path=video_path,
                                num_codes=num_codes,
                                camera=camera,
                                width=width,
                                height=height,
                                fps=fps,
                            )
                            if path:
                                rel_path = f"videos/code_{code_idx:03d}_conditional/clips_{ctx.clip_i:03d}_{ctx.clip_j:03d}.mp4"
                                cond_video_info.append({
                                    "path": rel_path,
                                    "clip_i": int(ctx.clip_i),
                                    "clip_j": int(ctx.clip_j),
                                    "succ_i": int(best_pair.succ_i),
                                    "succ_j": int(best_pair.succ_j),
                                    "qpos_dist": float(best_pair.qpos_distance),
                                })
                                cond_videos_rendered += 1
                        except Exception as e:
                            logging.warning(f"    Failed to render conditional video: {e}")

                    logging.info(f"    Rendered {len(cond_video_info)} conditional comparison videos")

                conditional_result = {
                    "image": cond_img_b64,
                    "avg_sim": float(conditional_data["avg_conditional_similarity"]),
                    "std_sim": float(conditional_data["std_conditional_similarity"]),
                    "total_matched": int(conditional_data["total_matched_frames"]),
                    "n_pairs": int(conditional_data["n_pairs"]),
                    "avg_qpos_distance": float(conditional_data["avg_qpos_distance"]),
                    "videos": cond_video_info,
                }
                logging.info(
                    f"    Conditional analysis: {conditional_data['n_pairs']} pairs, "
                    f"{conditional_data['total_matched_frames']} matched frames, "
                    f"sim={conditional_data['avg_conditional_similarity']:.3f}"
                )
            else:
                logging.info("    Conditional analysis: insufficient qpos-matched frames")

        code_analyses.append({
            "code_idx": int(code_idx),
            "image": img_b64,
            "stats": {
                "total_frames": int(total_count),
                "n_clips": int(n_clips),
                "avg_pred_sim": float(avg_pred_sim),
                "avg_succ_sim": float(avg_succ_sim),
                "avg_combined_sim": float(avg_combined_sim),
            },
            "videos": video_info,
            "conditional": conditional_result,
        })

    # Generate HTML (pass output_dir to resolve video paths for base64 embedding)
    html_path = generate_context_html(
        code_analyses,
        output_dir / "transition_context_analysis.html",
        output_dir=output_dir,
        title="Transition Context Analysis - Top Codes",
    )

    # Generate standalone conditional HTML if enabled
    conditional_html_path = None
    if conditional_cfg and conditional_cfg.get("enabled", False):
        conditional_html_path = generate_conditional_html(
            code_analyses,
            output_dir / "conditional_transition_analysis.html",
            output_dir=output_dir,
            title="Conditional Transition Analysis (qpos-matched)",
        )
        logging.info(f"  Conditional HTML viewer: {conditional_html_path}")

    # Save JSON summary
    json_summary = []
    for ca in code_analyses:
        entry = {
            "code_idx": ca["code_idx"],
            "stats": ca["stats"],
            "videos": ca["videos"],
        }
        # Include conditional stats if present
        if ca.get("conditional"):
            entry["conditional"] = {
                "avg_sim": ca["conditional"]["avg_sim"],
                "std_sim": ca["conditional"]["std_sim"],
                "total_matched": ca["conditional"]["total_matched"],
                "n_pairs": ca["conditional"]["n_pairs"],
                "avg_qpos_distance": ca["conditional"]["avg_qpos_distance"],
            }
        json_summary.append(entry)

    json_path = output_dir / "transition_context_stats.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)

    logging.info(f"\nTransition context analysis complete:")
    logging.info(f"  HTML viewer: {html_path}")
    logging.info(f"  JSON stats: {json_path}")

    return {
        "html_path": html_path,
        "conditional_html_path": conditional_html_path,
        "json_path": str(json_path),
        "code_analyses": code_analyses,
    }
