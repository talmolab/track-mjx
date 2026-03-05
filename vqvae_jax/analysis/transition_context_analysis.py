"""Transition Context Analysis for VQ-VAE Codes.

Provides:
- Global transition matrix computation and visualization
- Stationary distribution analysis
- Code popularity and pair popularity metrics
- Pose gallery rendering for popular codes
"""

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult
from .rendering import get_hierarchical_colormap, get_nature_colormap
from .utils import CodeRun, extract_code_runs


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


def compute_pair_popularity(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> dict[tuple[int, int], int]:
    """Compute total frame count for each (L0, L1) code pair across all clips.

    Args:
        results: List of InferenceResult from rollouts.
        num_codes: Number of codes per depth level.

    Returns:
        Dict mapping (l0_code, l1_code) to total frame count.
    """
    pair_counts: dict[tuple[int, int], int] = {}
    for result in results:
        if result.rvq_indices is None or len(result.rvq_indices) < 2:
            continue
        l0_codes = result.rvq_indices[0]
        l1_codes = result.rvq_indices[1]
        for t in range(len(l0_codes)):
            pair = (int(l0_codes[t]), int(l1_codes[t]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def compute_global_transition_matrix(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> tuple[np.ndarray, plt.Figure]:
    """Compute transition matrix aggregated across ALL clips.

    Args:
        results: List of InferenceResult from rollouts.
        num_codes: Total number of codes.

    Returns:
        Tuple of (transition_counts [num_codes, num_codes], matplotlib figure).
    """
    # Aggregate transition counts from all clips
    global_counts = np.zeros((num_codes, num_codes), dtype=np.int64)
    for result in results:
        indices = result.code_indices
        for i in range(len(indices) - 1):
            from_code = int(indices[i])
            to_code = int(indices[i + 1])
            global_counts[from_code, to_code] += 1

    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(12, 10))
    # Log scale for better visualization
    log_counts = np.log1p(global_counts)
    im = ax.imshow(log_counts, cmap="viridis", aspect="auto")
    ax.set_title(
        f"Global Transition Matrix (All {len(results)} Clips)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("To Code", fontsize=11)
    ax.set_ylabel("From Code", fontsize=11)
    plt.colorbar(im, ax=ax, label="log(count + 1)", shrink=0.8)

    # Add total transitions text
    total_transitions = global_counts.sum()
    ax.text(
        0.02,
        0.98,
        f"Total transitions: {total_transitions:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return global_counts, fig


def compute_stationary_distribution(
    transition_counts: np.ndarray,
    frame_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Compute stationary distribution and create comparison figure.

    Args:
        transition_counts: Raw transition count matrix [num_codes, num_codes].
        frame_counts: Optional dict mapping code_idx to total frame count.

    Returns:
        Dictionary with stationary_dist, empirical_dist, and figure.
    """
    num_codes = transition_counts.shape[0]

    # Row-normalize to get transition probability matrix
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    P = transition_counts / row_sums

    # Compute stationary distribution via eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()

    # Compute empirical distribution from frame counts
    empirical_dist = None
    if frame_counts is not None:
        total_frames = sum(frame_counts.values())
        empirical_dist = np.array(
            [frame_counts.get(i, 0) / total_frames for i in range(num_codes)]
        )

    # Create figure: Stationary vs Empirical Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(num_codes)
    width = 0.35

    ax.bar(
        x - width / 2,
        stationary,
        width,
        label="Stationary (theoretical)",
        color="steelblue",
        alpha=0.8,
    )
    if empirical_dist is not None:
        ax.bar(
            x + width / 2,
            empirical_dist,
            width,
            label="Empirical (observed)",
            color="coral",
            alpha=0.8,
        )

    ax.set_xlabel("Code Index", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Stationary vs Empirical Distribution", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(-1, num_codes)
    plt.tight_layout()

    return {
        "stationary_dist": stationary,
        "empirical_dist": empirical_dist,
        "figure": fig,
    }


def render_kinematic_profiles(
    results: Sequence[InferenceResult],
    code_idx: int,
    joint_names: list[str],
) -> plt.Figure:
    """Render kinematic profile line plots for clips using a specific code.

    Creates a 4-subplot figure showing mean +/- std across clips that use
    ``code_idx``:
    1. Root XY speed (norm of qvel[:, :2])
    2. Root Z height (qpos[:, 2])
    3. Left vs Right hip extension angles
    4. Left vs Right knee angles

    Args:
        results: Inference results with qpos and qvel.
        code_idx: D0 code to filter clips by.
        joint_names: Joint name list from walker config.

    Returns:
        Matplotlib figure with 4 subplots.
    """
    # Collect clips that use this code
    matching_clips = []
    for r in results:
        if code_idx in r.code_indices and r.qpos is not None and len(r.qpos) > 1:
            matching_clips.append(r)

    if len(matching_clips) < 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(
            0.5, 0.5, f"Code {code_idx}: < 2 clips", ha="center", va="center"
        )
        return fig

    # Resolve joint indices from names (qpos offset = 7 for root pos+quat)
    def _joint_qpos_idx(name: str) -> int | None:
        try:
            return joint_names.index(name) + 7
        except ValueError:
            return None

    hip_l_idx = _joint_qpos_idx("hip_L_extend")
    hip_r_idx = _joint_qpos_idx("hip_R_extend")
    knee_l_idx = _joint_qpos_idx("knee_L")
    knee_r_idx = _joint_qpos_idx("knee_R")

    # Truncate all clips to the minimum length for alignment
    min_len = min(len(r.qpos) for r in matching_clips)
    min_len = min(min_len, min(len(r.qvel) for r in matching_clips if r.qvel is not None))

    # Gather arrays
    xy_speeds = []
    z_heights = []
    hip_l_vals = []
    hip_r_vals = []
    knee_l_vals = []
    knee_r_vals = []

    for r in matching_clips:
        T = min_len
        z_heights.append(r.qpos[:T, 2])
        if r.qvel is not None and len(r.qvel) >= T:
            xy_speed = np.linalg.norm(r.qvel[:T, :2], axis=1)
            xy_speeds.append(xy_speed)
        if hip_l_idx is not None:
            hip_l_vals.append(r.qpos[:T, hip_l_idx])
        if hip_r_idx is not None:
            hip_r_vals.append(r.qpos[:T, hip_r_idx])
        if knee_l_idx is not None:
            knee_l_vals.append(r.qpos[:T, knee_l_idx])
        if knee_r_idx is not None:
            knee_r_vals.append(r.qpos[:T, knee_r_idx])

    t_axis = np.arange(min_len)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Kinematic Profiles for Code {code_idx} "
        f"({len(matching_clips)} clips)",
        fontsize=13,
        fontweight="bold",
    )

    def _plot_mean_std(ax, data_list, label, color):
        if not data_list:
            return
        arr = np.array(data_list)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(t_axis, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.2)

    # Panel 1: Root XY speed
    ax1 = axes[0, 0]
    _plot_mean_std(ax1, xy_speeds, "XY speed", "steelblue")
    ax1.set_ylabel("Root XY Speed")
    ax1.set_title("Root XY Speed")
    ax1.legend(fontsize=8)

    # Panel 2: Root Z height
    ax2 = axes[0, 1]
    _plot_mean_std(ax2, z_heights, "Z height", "coral")
    ax2.set_ylabel("Root Z (m)")
    ax2.set_title("Root Z Height")
    ax2.legend(fontsize=8)

    # Panel 3: Hip L vs R extension
    ax3 = axes[1, 0]
    _plot_mean_std(ax3, hip_l_vals, "hip_L_extend", "tab:blue")
    _plot_mean_std(ax3, hip_r_vals, "hip_R_extend", "tab:orange")
    ax3.set_ylabel("Angle (rad)")
    ax3.set_xlabel("Frame")
    ax3.set_title("Hip Extension (L vs R)")
    ax3.legend(fontsize=8)

    # Panel 4: Knee L vs R
    ax4 = axes[1, 1]
    _plot_mean_std(ax4, knee_l_vals, "knee_L", "tab:blue")
    _plot_mean_std(ax4, knee_r_vals, "knee_R", "tab:orange")
    ax4.set_ylabel("Angle (rad)")
    ax4.set_xlabel("Frame")
    ax4.set_title("Knee Angle (L vs R)")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    return fig


def render_code_pose_gallery(
    results: Sequence[InferenceResult],
    code_idx: int,
    num_codes: int,
    env: Any,
    output_path: Path,
    n_clips: int = 6,
    context_frames: int = 15,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    l1_code: int | None = None,
) -> str:
    """Render a grid video showing where a code starts across different clips.

    Creates a grid video (2x3 for 6 clips) showing synchronized playback of
    where the specified code first appears in different clips.

    Args:
        results: List of InferenceResult from rollouts.
        code_idx: The L0 code to find first occurrences of.
        num_codes: Total number of codes per depth level.
        env: Environment with mj_model attribute.
        output_path: Path to save video.
        n_clips: Number of clips to sample and show in grid.
        context_frames: Frames before/after code start to include.
        camera: Camera name.
        width: Total video width.
        height: Total video height.
        fps: Frames per second.
        l1_code: Optional L1 code. When provided, frames are matched on the
            exact (L0, L1) pair and the timeline uses hierarchical coloring.

    Returns:
        Path to rendered video, or empty string on failure.
    """
    import imageio
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find clips where this code (or pair) appears and get first occurrence
    clips_with_code = []
    for result in results:
        if result.qpos is None or len(result.qpos) == 0:
            continue
        l0_mask = result.code_indices == code_idx
        if (
            l1_code is not None
            and result.rvq_indices is not None
            and len(result.rvq_indices) >= 2
        ):
            l1_mask = result.rvq_indices[1] == l1_code
            frames = np.where(l0_mask & l1_mask)[0]
        else:
            frames = np.where(l0_mask)[0]
        if len(frames) > 0:
            first_frame = int(frames[0])
            clips_with_code.append((result, first_frame))

    if len(clips_with_code) < 2:
        logging.warning(f"Code {code_idx} appears in fewer than 2 clips")
        return ""

    # Sample n_clips clips evenly
    if len(clips_with_code) > n_clips:
        indices = np.linspace(0, len(clips_with_code) - 1, n_clips, dtype=int)
        clips_with_code = [clips_with_code[i] for i in indices]

    actual_clips = len(clips_with_code)

    # Determine grid layout (2 rows preferred)
    if actual_clips <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = 2, 3

    # Calculate cell dimensions
    cell_width = width // n_cols
    cell_height = height // n_rows
    bar_height = 30
    render_height = cell_height - bar_height

    # Choose colormap: hierarchical when rendering (L0, L1) pairs
    use_hierarchical = l1_code is not None
    if use_hierarchical:
        l0_colors, l1_colors_map = get_hierarchical_colormap(num_codes)
    code_colors = get_nature_colormap(num_codes)

    # Setup MuJoCo renderer
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=render_height, width=cell_width)

    # Get camera ID
    cam_id = -1
    if camera:
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        except Exception:
            pass

    # Compute frame ranges for each clip
    clip_ranges = []
    for result, first_frame in clips_with_code:
        start = max(0, first_frame - context_frames)
        end = min(len(result.qpos), first_frame + context_frames + 1)
        clip_ranges.append((result, first_frame, start, end))

    # Max frames across all clips
    max_frames = max(end - start for _, _, start, end in clip_ranges)

    if max_frames == 0:
        renderer.close()
        return ""

    frames_out = []
    for t in range(max_frames):
        # Create grid frame
        grid_frame = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background

        for cell_idx, (result, first_frame, start, end) in enumerate(clip_ranges):
            if cell_idx >= n_rows * n_cols:
                break

            row = cell_idx // n_cols
            col = cell_idx % n_cols
            y_offset = row * cell_height
            x_offset = col * cell_width

            # Get frame index for this clip (clamp to available range)
            n_clip_frames = end - start
            frame_idx = start + min(t, n_clip_frames - 1)

            # Render
            mj_data.qpos[:] = result.qpos[frame_idx]
            mujoco.mj_forward(mj_model, mj_data)
            if cam_id >= 0:
                renderer.update_scene(mj_data, camera=cam_id)
            else:
                renderer.update_scene(mj_data)
            cell_render = renderer.render()

            # Place render in grid
            grid_frame[
                y_offset : y_offset + render_height, x_offset : x_offset + cell_width
            ] = cell_render

            # Draw code timeline bar
            bar_y = y_offset + render_height
            for j in range(n_clip_frames):
                bx_start = x_offset + int(j * cell_width / n_clip_frames)
                bx_end = x_offset + int((j + 1) * cell_width / n_clip_frames)
                idx = start + j
                if idx < len(result.code_indices):
                    c_idx = int(result.code_indices[idx])
                    # Get color: hierarchical (L0,L1) or flat L0
                    if (
                        use_hierarchical
                        and result.rvq_indices is not None
                        and len(result.rvq_indices) >= 2
                        and idx < len(result.rvq_indices[1])
                    ):
                        c1_idx = int(result.rvq_indices[1][idx])
                        color = l1_colors_map[c_idx, c1_idx]
                    else:
                        color = code_colors[c_idx]
                    # Highlight: exact (L0, L1) pair match (or just L0 if no L1)
                    is_target = c_idx == code_idx
                    if (
                        is_target
                        and use_hierarchical
                        and result.rvq_indices is not None
                        and len(result.rvq_indices) >= 2
                        and idx < len(result.rvq_indices[1])
                    ):
                        is_target = int(result.rvq_indices[1][idx]) == l1_code
                    if is_target:
                        # Highlight target with white border
                        grid_frame[bar_y : bar_y + 2, bx_start:bx_end] = [
                            255,
                            255,
                            255,
                        ]
                        grid_frame[
                            bar_y + bar_height - 2 : bar_y + bar_height,
                            bx_start:bx_end,
                        ] = [255, 255, 255]
                        grid_frame[
                            bar_y + 2 : bar_y + bar_height - 2, bx_start:bx_end
                        ] = color
                    else:
                        grid_frame[bar_y : bar_y + bar_height, bx_start:bx_end] = color

            # Playhead
            if n_clip_frames > 0:
                playhead_x = x_offset + int(
                    min(t, n_clip_frames - 1) * cell_width / n_clip_frames
                )
                grid_frame[bar_y : bar_y + bar_height, playhead_x : playhead_x + 2] = [
                    255,
                    255,
                    255,
                ]

        frames_out.append(grid_frame)

    renderer.close()

    if len(frames_out) == 0:
        return ""

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames_out:
            writer.append_data(frame)

    return str(output_path)


def compute_transition_ngram_popularity(
    results: Sequence[InferenceResult],
    num_codes: int,
    n: int = 3,
) -> dict[tuple[int, ...], int]:
    """Count occurrences of n-length code-run transitions across all clips.

    Slides an n-length window over the code runs of each clip and counts
    how many times each unique n-gram (tuple of consecutive codes) appears.

    Args:
        results: Inference results with code_indices populated.
        num_codes: Total number of codes (unused, kept for API consistency).
        n: Length of transition n-gram (number of consecutive code runs).

    Returns:
        Dict mapping n-gram tuple to occurrence count.
    """
    ngram_counts: dict[tuple[int, ...], int] = {}
    for result in results:
        runs = extract_code_runs(result.code_indices)
        if len(runs) < n:
            continue
        for i in range(len(runs) - n + 1):
            ngram = tuple(run.code for run in runs[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts


def get_top_k_transitions(
    ngram_counts: dict[tuple[int, ...], int],
    top_k: int,
) -> list[tuple[tuple[int, ...], int]]:
    """Get top-K most popular n-gram transitions by count.

    Args:
        ngram_counts: Dict mapping n-gram tuple to count.
        top_k: Number of top transitions to return.

    Returns:
        List of (ngram_tuple, count) sorted by count descending.
    """
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_ngrams[:top_k]


def render_transition_pose_gallery(
    results: Sequence[InferenceResult],
    transition: tuple[int, ...],
    num_codes: int,
    env: Any,
    output_path: Path,
    n_clips: int = 4,
    camera: str | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
) -> str:
    """Render a grid video showing clips that exhibit a specific code transition.

    For each matching clip, renders exactly the frames spanning the n consecutive
    code runs that form the transition — no padding before or after.

    Args:
        results: List of InferenceResult from rollouts.
        transition: Tuple of code indices defining the transition n-gram.
        num_codes: Total number of codes per depth level.
        env: Environment with mj_model attribute.
        output_path: Path to save video.
        n_clips: Number of clips to sample and show in grid.
        camera: Camera name.
        width: Total video width.
        height: Total video height.
        fps: Frames per second.

    Returns:
        Path to rendered video, or empty string on failure.
    """
    import imageio
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(transition)

    # Find all clips containing this transition
    # Each match: (result, start_frame, end_frame)
    matches: list[tuple[InferenceResult, int, int]] = []
    for result in results:
        if result.qpos is None or len(result.qpos) == 0:
            continue
        runs = extract_code_runs(result.code_indices)
        if len(runs) < n:
            continue
        for i in range(len(runs) - n + 1):
            ngram = tuple(run.code for run in runs[i : i + n])
            if ngram == transition:
                start_frame = runs[i].start_frame
                end_frame = runs[i + n - 1].end_frame
                matches.append((result, start_frame, end_frame))

    if len(matches) < 2:
        logging.warning(
            f"Transition {transition} found in fewer than 2 clips/locations"
        )
        return ""

    # Sample n_clips matches evenly
    if len(matches) > n_clips:
        indices = np.linspace(0, len(matches) - 1, n_clips, dtype=int)
        matches = [matches[i] for i in indices]

    actual_clips = len(matches)

    # Determine grid layout (2 rows)
    if actual_clips <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = 2, 3

    # Calculate cell dimensions
    cell_width = width // n_cols
    cell_height = height // n_rows
    bar_height = 30
    render_height = cell_height - bar_height

    code_colors = get_nature_colormap(num_codes)
    transition_codes = set(transition)

    # Setup MuJoCo renderer
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=render_height, width=cell_width)

    # Get camera ID
    cam_id = -1
    if camera:
        try:
            cam_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
        except Exception:
            pass

    # Max frames across all matches
    max_frames = max(end - start for _, start, end in matches)
    if max_frames == 0:
        renderer.close()
        return ""

    frames_out = []
    for t in range(max_frames):
        grid_frame = np.ones((height, width, 3), dtype=np.uint8) * 30

        for cell_idx, (result, start, end) in enumerate(matches):
            if cell_idx >= n_rows * n_cols:
                break

            row = cell_idx // n_cols
            col = cell_idx % n_cols
            y_offset = row * cell_height
            x_offset = col * cell_width

            n_clip_frames = end - start
            frame_idx = start + min(t, n_clip_frames - 1)

            # Render pose
            mj_data.qpos[:] = result.qpos[frame_idx]
            mujoco.mj_forward(mj_model, mj_data)
            if cam_id >= 0:
                renderer.update_scene(mj_data, camera=cam_id)
            else:
                renderer.update_scene(mj_data)
            cell_render = renderer.render()

            grid_frame[
                y_offset : y_offset + render_height,
                x_offset : x_offset + cell_width,
            ] = cell_render

            # Draw code timeline bar
            bar_y = y_offset + render_height
            for j in range(n_clip_frames):
                bx_start = x_offset + int(j * cell_width / n_clip_frames)
                bx_end = x_offset + int((j + 1) * cell_width / n_clip_frames)
                idx = start + j
                if idx < len(result.code_indices):
                    c_idx = int(result.code_indices[idx])
                    color = code_colors[c_idx]
                    # Highlight codes that are part of the transition
                    if c_idx in transition_codes:
                        grid_frame[bar_y : bar_y + 2, bx_start:bx_end] = [
                            255, 255, 255,
                        ]
                        grid_frame[
                            bar_y + bar_height - 2 : bar_y + bar_height,
                            bx_start:bx_end,
                        ] = [255, 255, 255]
                        grid_frame[
                            bar_y + 2 : bar_y + bar_height - 2,
                            bx_start:bx_end,
                        ] = color
                    else:
                        grid_frame[
                            bar_y : bar_y + bar_height, bx_start:bx_end
                        ] = color

            # Playhead
            if n_clip_frames > 0:
                playhead_x = x_offset + int(
                    min(t, n_clip_frames - 1) * cell_width / n_clip_frames
                )
                grid_frame[
                    bar_y : bar_y + bar_height, playhead_x : playhead_x + 2
                ] = [255, 255, 255]

        frames_out.append(grid_frame)

    renderer.close()

    if len(frames_out) == 0:
        return ""

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames_out:
            writer.append_data(frame)

    return str(output_path)
