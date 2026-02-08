"""t-SNE Skill-Space Trajectory Visualization.

Visualizes how different clips traverse the VQ-VAE "skill space" over time by:

1. Selecting high-movement clips (root XYZ displacement, including Z for rearing)
2. Treating k-transition sequences as points (concatenated codebook vectors)
3. Running t-SNE to embed all points in 2D
4. Creating a synchronized HTML viewer: animated t-SNE canvas + video playback
5. Supporting n_clips comparison in the same t-SNE space with different colors

Key concepts:
- Movement score: Total root XYZ displacement over all frames (captures both
  locomotion and rearing behaviors).
- k-transition point: A sliding window of k+1 code runs, represented as the
  concatenation of their codebook vectors. For k=8 with latent_dim=32 this
  gives a 288-dimensional feature vector per point.
- Synchronized viewer: HTML page with a t-SNE canvas on the left and a video
  player on the right. The canvas highlights the active k-transition point as
  the video plays, with trails connecting recent points.
"""

import base64
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .compositional_transition_analysis import extract_code_runs, CodeRun
from .inference_cache import InferenceResult


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class KTransition:
    """A single k-transition point for t-SNE embedding.

    Attributes:
        clip_idx: Index of the source clip.
        start_frame: First frame of the k-transition.
        end_frame: Last frame (exclusive).
        midpoint_frame: Middle frame (for syncing with video).
        code_sequence: Tuple of k+1 codes in the transition.
        embedding: Concatenated codebook vectors, shape [(k+1)*latent_dim].
    """

    clip_idx: int
    start_frame: int
    end_frame: int
    midpoint_frame: int
    code_sequence: tuple[int, ...]
    embedding: np.ndarray


@dataclass
class ClipTrajectoryData:
    """All t-SNE data for one clip.

    Attributes:
        clip_idx: Index of the source clip.
        result: The InferenceResult for this clip.
        transitions: List of KTransition points extracted from this clip.
        total_movement: Root XYZ displacement for ranking.
        avg_frames_per_transition: Average frame span per k-transition.
    """

    clip_idx: int
    result: InferenceResult
    transitions: list[KTransition]
    total_movement: float
    avg_frames_per_transition: float


class MovementCategory(Enum):
    """Movement category for t-SNE trajectory clip selection.

    Attributes:
        HIGH_XYZ: Top clips by total XYZ displacement.
        HIGH_XY: High XY movement, low Z (locomotion without rearing).
        HIGH_Z: High Z movement, low XY (rearing/vertical in place).
        LOW_XYZ: Lowest total XYZ movement (stationary).
        XY_VS_XYZ: HIGH_XY + HIGH_XYZ clips in the same t-SNE space.
    """

    HIGH_XYZ = "high_xyz"
    HIGH_XY = "high_xy"
    HIGH_Z = "high_z"
    LOW_XYZ = "low_xyz"
    XY_VS_XYZ = "xy_vs_xyz"


# Labels and descriptions for each category (for UI display)
_CATEGORY_META: dict[MovementCategory, tuple[str, str]] = {
    MovementCategory.HIGH_XYZ: (
        "High XYZ",
        "Top clips by total XYZ displacement",
    ),
    MovementCategory.HIGH_XY: (
        "High XY",
        "High XY movement, low Z (locomotion)",
    ),
    MovementCategory.HIGH_Z: (
        "High Z",
        "High Z movement, low XY (rearing)",
    ),
    MovementCategory.LOW_XYZ: (
        "Low XYZ",
        "Lowest total XYZ movement (stationary)",
    ),
    MovementCategory.XY_VS_XYZ: (
        "XY vs XYZ",
        "HIGH_XY (blue, locomotion) + HIGH_XYZ (red, locomotion+rearing)",
    ),
}

# Group colors for XY_VS_XYZ category
_GROUP_COLORS: dict[str, str] = {
    "HIGH_XY": "#1f77b4",  # blue
    "HIGH_XYZ": "#d62728",  # red
}


@dataclass
class CategoryData:
    """All data for one movement category in the multicategory viewer.

    Attributes:
        category: The movement category enum value.
        label: Human-readable label for the category.
        description: Short description of what this category captures.
        clip_data_list: List of ClipTrajectoryData for clips in this category.
        group_labels: Optional mapping from clip_idx to group name (e.g.
            "HIGH_XY", "HIGH_XYZ") for categories with sub-groups.
    """

    category: MovementCategory
    label: str
    description: str
    clip_data_list: list[ClipTrajectoryData]
    group_labels: dict[int, str] | None = None


# =============================================================================
# CLIP SELECTION
# =============================================================================


def compute_clip_movement(result: InferenceResult) -> float:
    """Compute total root XYZ displacement for a clip.

    Sums the Euclidean distance of frame-to-frame root position changes,
    capturing both horizontal locomotion (XY) and rearing (Z).

    Args:
        result: InferenceResult for one clip.

    Returns:
        Total displacement in meters.
    """
    root_pos = result.qpos[:, :3]  # [T, 3] - xyz
    displacements = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    return float(np.sum(displacements))


def select_clips_by_movement(
    results: Sequence[InferenceResult],
    n_clips: int,
) -> list[InferenceResult]:
    """Select the top-N clips by total root XYZ movement.

    Args:
        results: All InferenceResult objects.
        n_clips: Number of clips to select.

    Returns:
        List of InferenceResult objects sorted by movement (descending).
    """
    scored = [(compute_clip_movement(r), r) for r in results]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = scored[:n_clips]
    for movement, result in selected:
        logging.info(f"  Selected clip {result.clip_idx}: movement={movement:.4f}")

    return [r for _, r in selected]


def compute_clip_xy_movement(result: InferenceResult) -> float:
    """Compute total root XY-plane displacement for a clip.

    Args:
        result: InferenceResult for one clip.

    Returns:
        Total XY displacement in meters.
    """
    root_xy = result.qpos[:, :2]  # [T, 2] - xy only
    displacements = np.linalg.norm(np.diff(root_xy, axis=0), axis=1)
    return float(np.sum(displacements))


def compute_clip_z_movement(result: InferenceResult) -> float:
    """Compute total absolute root Z displacement for a clip.

    Args:
        result: InferenceResult for one clip.

    Returns:
        Total absolute Z displacement in meters.
    """
    root_z = result.qpos[:, 2]  # [T] - z only
    displacements = np.abs(np.diff(root_z))
    return float(np.sum(displacements))


def _compute_movement_profile(result: InferenceResult) -> np.ndarray:
    """Compute a 5D movement profile vector for a clip.

    Returns:
        Array of [total_xy, total_z, mean_speed, speed_std, z_range].
    """
    root_pos = result.qpos[:, :3]
    diffs = np.diff(root_pos, axis=0)  # [T-1, 3]
    speeds = np.linalg.norm(diffs, axis=1)
    total_xy = float(np.sum(np.linalg.norm(diffs[:, :2], axis=1)))
    total_z = float(np.sum(np.abs(diffs[:, 2])))
    mean_speed = float(np.mean(speeds))
    speed_std = float(np.std(speeds))
    z_range = float(np.ptp(root_pos[:, 2]))
    return np.array([total_xy, total_z, mean_speed, speed_std, z_range])


def _select_similar_clips(
    candidates: list[InferenceResult],
    n_select: int,
) -> list[InferenceResult]:
    """Greedily select n_select clips with the tightest movement profiles.

    Computes movement profiles, z-score normalizes, then greedily grows a
    cluster starting from the candidate with the lowest mean pairwise distance.

    Args:
        candidates: Pool of candidate clips (should be >= n_select).
        n_select: Number of clips to select.

    Returns:
        Selected clips.
    """
    if len(candidates) <= n_select:
        return candidates

    profiles = np.stack([_compute_movement_profile(r) for r in candidates])
    # Z-score normalize
    mu = profiles.mean(axis=0)
    sigma = profiles.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    normed = (profiles - mu) / sigma

    # Pairwise Euclidean distance
    n = len(normed)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(normed[i] - normed[j])
            dists[i, j] = d
            dists[j, i] = d

    # Seed: candidate with lowest mean distance to all others
    mean_dists = dists.mean(axis=1)
    seed = int(np.argmin(mean_dists))

    selected_idxs = [seed]
    remaining = set(range(n)) - {seed}

    while len(selected_idxs) < n_select and remaining:
        # Pick the remaining candidate closest to the current cluster centroid
        best_idx = -1
        best_dist = float("inf")
        for r in remaining:
            avg_d = np.mean([dists[r, s] for s in selected_idxs])
            if avg_d < best_dist:
                best_dist = avg_d
                best_idx = r
        selected_idxs.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected_idxs]


def select_clips_by_category(
    results: Sequence[InferenceResult],
    n_clips: int,
    xy_vs_xyz_cfg: dict[str, Any] | None = None,
) -> dict[MovementCategory, list[InferenceResult]]:
    """Select clips for each movement category.

    Args:
        results: All InferenceResult objects.
        n_clips: Number of clips to select per category.
        xy_vs_xyz_cfg: Config for XY_VS_XYZ category with key
            ``n_clips_per_group`` (default 3).

    Returns:
        Dict mapping each MovementCategory to its selected clips.
    """
    # Pre-compute scores for all clips
    scores = []
    for r in results:
        scores.append(
            {
                "result": r,
                "xyz": compute_clip_movement(r),
                "xy": compute_clip_xy_movement(r),
                "z": compute_clip_z_movement(r),
            }
        )

    # Compute medians for filtering
    all_z = [s["z"] for s in scores]
    all_xy = [s["xy"] for s in scores]
    median_z = float(np.median(all_z))
    median_xy = float(np.median(all_xy))

    categories: dict[MovementCategory, list[InferenceResult]] = {}

    # HIGH_XYZ: top n_clips by total XYZ movement
    by_xyz = sorted(scores, key=lambda s: s["xyz"], reverse=True)
    categories[MovementCategory.HIGH_XYZ] = [s["result"] for s in by_xyz[:n_clips]]

    # HIGH_XY: filter to clips with Z below median, then top by XY
    low_z_pool = [s for s in scores if s["z"] <= median_z]
    low_z_pool.sort(key=lambda s: s["xy"], reverse=True)
    categories[MovementCategory.HIGH_XY] = [s["result"] for s in low_z_pool[:n_clips]]

    # HIGH_Z: filter to clips with XY below median, then top by Z
    low_xy_pool = [s for s in scores if s["xy"] <= median_xy]
    low_xy_pool.sort(key=lambda s: s["z"], reverse=True)
    categories[MovementCategory.HIGH_Z] = [s["result"] for s in low_xy_pool[:n_clips]]

    # LOW_XYZ: bottom n_clips by total XYZ movement
    by_xyz_asc = sorted(scores, key=lambda s: s["xyz"])
    categories[MovementCategory.LOW_XYZ] = [s["result"] for s in by_xyz_asc[:n_clips]]

    # XY_VS_XYZ: similar HIGH_XY clips + similar HIGH_XYZ clips
    xy_vs_xyz_cfg = xy_vs_xyz_cfg or {}
    n_per_group = xy_vs_xyz_cfg.get("n_clips_per_group", 3)
    candidate_mult = 3  # Take 3x candidates for similarity filtering

    # XY group: low Z, high XY -> select similar subset
    xy_candidates = low_z_pool[: n_per_group * candidate_mult]
    xy_candidate_results = [s["result"] for s in xy_candidates]
    xy_selected = _select_similar_clips(xy_candidate_results, n_per_group)

    # XYZ group: top by XYZ -> select similar subset
    xyz_candidates = by_xyz[: n_per_group * candidate_mult]
    xyz_candidate_results = [s["result"] for s in xyz_candidates]
    xyz_selected = _select_similar_clips(xyz_candidate_results, n_per_group)

    categories[MovementCategory.XY_VS_XYZ] = xy_selected + xyz_selected

    # Log category selections
    for cat, clips in categories.items():
        clip_idxs = [r.clip_idx for r in clips]
        logging.info(f"  {cat.value}: {len(clips)} clips -> {clip_idxs}")

    return categories


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def extract_k_transitions_for_clip(
    result: InferenceResult,
    k: int,
    codebook: np.ndarray,
) -> list[KTransition]:
    """Extract k-transition points from a single clip.

    Slides a window of size k+1 over the code runs and creates a feature
    vector by concatenating the codebook vectors for each code in the window.

    Args:
        result: InferenceResult for one clip.
        k: Number of transitions (window will contain k+1 codes).
        codebook: Codebook array of shape [num_codes, latent_dim].

    Returns:
        List of KTransition objects.
    """
    runs = extract_code_runs(result.code_indices)
    window_size = k + 1

    if len(runs) < window_size:
        return []

    transitions = []
    for i in range(len(runs) - window_size + 1):
        sub_runs = runs[i : i + window_size]
        code_seq = tuple(r.code for r in sub_runs)
        start_frame = sub_runs[0].start_frame
        end_frame = sub_runs[-1].end_frame
        midpoint_frame = (start_frame + end_frame) // 2

        # Concatenate codebook vectors for each code in the window
        embedding = np.concatenate([codebook[c] for c in code_seq])

        transitions.append(
            KTransition(
                clip_idx=result.clip_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                midpoint_frame=midpoint_frame,
                code_sequence=code_seq,
                embedding=embedding,
            )
        )

    return transitions


def compute_tsne_embedding(
    all_transitions: list[KTransition],
    perplexity: float = 30.0,
) -> np.ndarray | None:
    """Run t-SNE on all k-transition embeddings.

    Args:
        all_transitions: Combined list of KTransition from all clips.
        perplexity: t-SNE perplexity parameter.

    Returns:
        2D coordinates of shape [N_total, 2], or None if too few samples.
    """
    from sklearn.manifold import TSNE

    embeddings = np.stack([t.embedding for t in all_transitions])
    n_samples = len(embeddings)

    if n_samples < 4:
        logging.warning(f"  Too few samples ({n_samples}) for t-SNE, skipping")
        return None

    # Perplexity must be < n_samples; use at most (n_samples - 1) / 3
    effective_perplexity = min(perplexity, (n_samples - 1) / 3.0)
    effective_perplexity = max(2.0, effective_perplexity)
    if effective_perplexity != perplexity:
        logging.info(
            f"  Adjusted t-SNE perplexity from {perplexity} to "
            f"{effective_perplexity} for {n_samples} samples"
        )

    logging.info(
        f"  Running t-SNE on {n_samples} points "
        f"(dim={embeddings.shape[1]}, perplexity={effective_perplexity:.1f})..."
    )

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
    )
    coords = tsne.fit_transform(embeddings)

    logging.info(
        f"  t-SNE complete. Range: x=[{coords[:, 0].min():.1f}, "
        f"{coords[:, 0].max():.1f}], y=[{coords[:, 1].min():.1f}, "
        f"{coords[:, 1].max():.1f}]"
    )

    return coords


def compute_umap_embedding(
    all_transitions: list[KTransition],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray | None:
    """Run UMAP on all k-transition embeddings.

    Args:
        all_transitions: Combined list of KTransition from all clips.
        n_neighbors: UMAP n_neighbors parameter (larger = more global).
        min_dist: UMAP min_dist parameter (smaller = tighter clusters).

    Returns:
        2D coordinates of shape [N_total, 2], or None if too few samples.
    """
    import umap

    embeddings = np.stack([t.embedding for t in all_transitions])
    n_samples = len(embeddings)

    if n_samples < 4:
        logging.warning(f"  Too few samples ({n_samples}) for UMAP, skipping")
        return None

    effective_n_neighbors = min(n_neighbors, n_samples - 1)
    if effective_n_neighbors != n_neighbors:
        logging.info(
            f"  Adjusted UMAP n_neighbors from {n_neighbors} to "
            f"{effective_n_neighbors} for {n_samples} samples"
        )

    logging.info(
        f"  Running UMAP on {n_samples} points "
        f"(dim={embeddings.shape[1]}, n_neighbors={effective_n_neighbors}, "
        f"min_dist={min_dist})..."
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    logging.info(
        f"  UMAP complete. Range: x=[{coords[:, 0].min():.1f}, "
        f"{coords[:, 0].max():.1f}], y=[{coords[:, 1].min():.1f}, "
        f"{coords[:, 1].max():.1f}]"
    )

    return coords


# =============================================================================
# VIDEO RENDERING
# =============================================================================


def render_clip_video_for_tsne(
    env: Any,
    result: InferenceResult,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> str:
    """Render a full clip to base64-encoded MP4 video.

    Uses qpos-based MuJoCo rendering with a code timeline bar and clip label
    overlay.

    Args:
        env: Environment with mj_model for rendering.
        result: InferenceResult containing clip data.
        camera: Camera name for rendering.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.

    Returns:
        Base64 data URL string for the video.
    """
    import imageio
    import mujoco

    from .rendering import add_text_overlay, get_nature_colormap

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    bar_height = 30
    render_height = height - bar_height
    renderer = mujoco.Renderer(mj_model, height=render_height, width=width)

    # Get camera ID
    cam_id = -1
    if camera:
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        except Exception:
            pass

    num_codes = int(np.max(result.code_indices)) + 1
    code_colors = get_nature_colormap(num_codes)
    n_frames = len(result.qpos)
    clip_label = f"Clip {result.clip_idx}"

    frames = []
    for i in range(n_frames):
        mj_data.qpos[:] = result.qpos[i]
        mujoco.mj_forward(mj_model, mj_data)

        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)
        render_frame = renderer.render().copy()

        # Add clip label overlay
        label_x = max(width - 120, 10)
        render_frame = add_text_overlay(
            render_frame,
            clip_label,
            position=(label_x, 8),
            font_size=16,
        )

        # Build full frame with code timeline bar
        full_frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        full_frame[:render_height, :] = render_frame

        # Code timeline bar
        for j in range(n_frames):
            x_start = int(j * width / n_frames)
            x_end = int((j + 1) * width / n_frames)
            code_idx = int(result.code_indices[j])
            full_frame[
                render_height : render_height + bar_height - 2,
                x_start:x_end,
            ] = code_colors[code_idx]

        # Playhead
        px = int(i * width / n_frames)
        full_frame[render_height : render_height + bar_height - 2, px : px + 2] = [
            255,
            255,
            255,
        ]

        frames.append(full_frame)

    renderer.close()

    # Write to temporary file and convert to base64
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    with imageio.get_writer(tmp_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    with open(tmp_path, "rb") as f:
        video_data = f.read()
    b64 = base64.b64encode(video_data).decode("utf-8")

    Path(tmp_path).unlink(missing_ok=True)

    return f"data:video/mp4;base64,{b64}"


# =============================================================================
# HTML VISUALIZATION
# =============================================================================

# Qualitative color palette (tab10-inspired, as hex)
_CLIP_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def generate_tsne_trajectory_html(
    clip_data_list: list[ClipTrajectoryData],
    tsne_coords: np.ndarray,
    videos_b64: list[str],
    k: int,
    fps: int,
    output_path: str | Path,
) -> str:
    """Generate the synchronized t-SNE trajectory HTML viewer.

    Creates an HTML page with a t-SNE canvas on top and all clip videos
    side-by-side below. All videos play in sync; clicking a video highlights
    that clip's trail on the canvas.

    Args:
        clip_data_list: List of ClipTrajectoryData for each clip.
        tsne_coords: 2D t-SNE coordinates, shape [N_total, 2].
        videos_b64: List of base64 data URLs for each clip's video.
        k: Number of transitions per point.
        fps: Video frames per second.
        output_path: Path to save the HTML file.

    Returns:
        Path to the saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-clip point arrays for JS
    clip_js_data = []
    point_idx = 0
    for clip_data in clip_data_list:
        points = []
        for t in clip_data.transitions:
            x = float(tsne_coords[point_idx, 0])
            y = float(tsne_coords[point_idx, 1])
            points.append(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "sf": t.start_frame,
                    "ef": t.end_frame,
                    "mf": t.midpoint_frame,
                    "codes": list(t.code_sequence),
                }
            )
            point_idx += 1

        clip_js_data.append(
            {
                "clipIdx": clip_data.clip_idx,
                "color": _CLIP_COLORS[len(clip_js_data) % len(_CLIP_COLORS)],
                "movement": round(clip_data.total_movement, 4),
                "avgFrames": round(clip_data.avg_frames_per_transition, 1),
                "points": points,
            }
        )

    # Compute t-SNE bounds for canvas scaling
    x_min, x_max = float(tsne_coords[:, 0].min()), float(tsne_coords[:, 0].max())
    y_min, y_max = float(tsne_coords[:, 1].min()), float(tsne_coords[:, 1].max())
    # Add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin = 0.08
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin

    # Build video sources JSON
    video_sources = json.dumps(videos_b64)
    clips_json = json.dumps(clip_js_data)

    # Stats summary
    total_points = sum(len(cd.transitions) for cd in clip_data_list)
    n_clips = len(clip_data_list)

    # Build video elements HTML (one per clip, side by side)
    video_cells_html = ""
    for i, cd in enumerate(clip_data_list):
        color = _CLIP_COLORS[i % len(_CLIP_COLORS)]
        video_cells_html += (
            f'<div class="video-cell" data-clip-idx="{i}" '
            f'onclick="setActiveClip({i})">\n'
            f'  <div class="video-label" style="color:{color}">'
            f"Clip {cd.clip_idx}"
            f' <span class="movement-tag">'
            f"mvmt={cd.total_movement:.3f}</span></div>\n"
            f'  <video id="vid{i}" loop muted playsinline'
            f' src="{videos_b64[i] if i < len(videos_b64) else ""}">'
            f"</video>\n"
            f"</div>\n"
        )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>t-SNE Skill-Space Trajectory Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 16px;
}}
h1 {{
    text-align: center;
    font-size: 1.6em;
    margin-bottom: 12px;
    color: #64b5f6;
}}
.outer-container {{
    max-width: 1400px;
    margin: 0 auto;
}}
.canvas-panel {{
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 16px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
    flex-wrap: wrap;
}}
.canvas-wrapper {{
    flex: 0 0 auto;
}}
.canvas-sidebar {{
    flex: 1;
    min-width: 200px;
}}
canvas {{
    display: block;
    border-radius: 8px;
    background: #ffffff;
    cursor: crosshair;
}}
.video-row {{
    display: flex;
    gap: 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 12px;
}}
.video-cell {{
    flex: 1;
    min-width: 0;
    cursor: pointer;
    border-radius: 10px;
    border: 3px solid transparent;
    padding: 6px;
    transition: border-color 0.2s, background 0.2s;
}}
.video-cell:hover {{
    background: rgba(255,255,255,0.04);
}}
.video-cell.active {{
    border-color: currentColor;
    background: rgba(255,255,255,0.06);
}}
.video-label {{
    font-size: 0.9em;
    font-weight: 600;
    margin-bottom: 6px;
    text-align: center;
}}
.movement-tag {{
    font-size: 0.75em;
    font-weight: 400;
    opacity: 0.7;
}}
.video-cell video {{
    width: 100%;
    border-radius: 8px;
    background: #000;
    display: block;
}}
.controls-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 10px 16px;
    border: 1px solid rgba(100,181,246,0.2);
}}
.controls-bar button {{
    padding: 5px 14px;
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    color: #90caf9;
    cursor: pointer;
    font-size: 0.85em;
}}
.controls-bar button:hover {{ background: rgba(100,181,246,0.2); }}
.controls-bar button.active {{
    background: rgba(100,181,246,0.3);
    border-color: #64b5f6;
    color: #fff;
}}
.stats {{
    font-size: 0.8em;
    color: #888;
    line-height: 1.6;
}}
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
    padding: 8px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.8em;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.info-box {{
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    font-size: 0.8em;
    color: #aaa;
    min-height: 36px;
}}
.spacer {{ flex: 1; }}
</style>
</head>
<body>
<h1>t-SNE Skill-Space Trajectory Viewer</h1>
<div class="outer-container">

  <div class="canvas-panel">
    <div class="canvas-wrapper">
      <canvas id="tsneCanvas" width="576" height="576"></canvas>
    </div>
    <div class="canvas-sidebar">
      <div class="legend" id="legend"></div>
      <div class="info-box" id="infoBox">Press Play to begin.</div>
      <div class="stats">
        {n_clips} clips | {total_points} t-SNE points | k={k} transitions
      </div>
    </div>
  </div>

  <div class="video-row">
    {video_cells_html}
  </div>

  <div class="controls-bar">
    <button id="playPauseBtn" onclick="togglePlay()">Play All</button>
    <button class="active" data-speed="1" onclick="setSpeed(1, this)">1x</button>
    <button data-speed="0.5" onclick="setSpeed(0.5, this)">0.5x</button>
    <button data-speed="2" onclick="setSpeed(2, this)">2x</button>
    <span class="spacer"></span>
    <span class="stats">Click a video to highlight its trail on the canvas</span>
  </div>

</div>

<script>
// === DATA ===
var clips = {clips_json};
var videoSources = {video_sources};
var FPS = {fps};
var nClips = {n_clips};
var xMin = {x_min}, xMax = {x_max}, yMin = {y_min}, yMax = {y_max};

// === STATE ===
var activeClipIdx = 0;
var currentFrame = 0;
var trailLength = 12;

// === DOM REFS ===
var canvas = document.getElementById('tsneCanvas');
var ctx = canvas.getContext('2d');
var W = canvas.width, H = canvas.height;
var infoBox = document.getElementById('infoBox');

// Collect all video elements
var videos = [];
for (var i = 0; i < nClips; i++) {{
    videos.push(document.getElementById('vid' + i));
}}

// === COORDINATE TRANSFORM ===
function tsneToCanvas(x, y) {{
    var cx = ((x - xMin) / (xMax - xMin)) * (W - 40) + 20;
    var cy = ((y - yMin) / (yMax - yMin)) * (H - 40) + 20;
    cy = H - cy;
    return [cx, cy];
}}

// === FIND ACTIVE POINT ===
function findActivePoint(points, frame) {{
    var bestIdx = -1;
    for (var i = 0; i < points.length; i++) {{
        if (frame >= points[i].sf && frame < points[i].ef) {{
            return i;
        }}
        if (frame >= points[i].ef) {{
            bestIdx = i;
        }}
    }}
    if (bestIdx === -1 && points.length > 0) {{
        bestIdx = 0;
    }}
    return bestIdx;
}}

// === DRAWING ===
var pulsePhase = 0;

function drawCanvas() {{
    ctx.clearRect(0, 0, W, H);

    // Draw grid
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 1;
    for (var gx = 0; gx < W; gx += 50) {{
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (var gy = 0; gy < H; gy += 50) {{
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    pulsePhase += 0.08;
    var pulseSize = 3 + Math.sin(pulsePhase) * 2;

    // Draw all clips (non-active first, active last for z-order)
    var drawOrder = [];
    for (var ci = 0; ci < clips.length; ci++) {{
        if (ci !== activeClipIdx) drawOrder.push(ci);
    }}
    drawOrder.push(activeClipIdx);

    for (var di = 0; di < drawOrder.length; di++) {{
        var ci = drawOrder[di];
        var clip = clips[ci];
        var isActive = (ci === activeClipIdx);
        var color = clip.color;
        var points = clip.points;
        var activeIdx = findActivePoint(points, currentFrame);

        // Draw all points as small dots
        for (var pi = 0; pi < points.length; pi++) {{
            var p = tsneToCanvas(points[pi].x, points[pi].y);
            ctx.beginPath();
            ctx.arc(p[0], p[1], isActive ? 3 : 2, 0, Math.PI * 2);
            ctx.fillStyle = isActive
                ? hexToRGBA(color, 0.4)
                : hexToRGBA(color, 0.15);
            ctx.fill();
        }}

        if (activeIdx < 0) continue;
        var trailStart = Math.max(0, activeIdx - trailLength);

        if (isActive) {{
            // Bright trail
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = hexToRGBA(color, 0.7);
            ctx.beginPath();
            for (var ti = trailStart; ti <= activeIdx; ti++) {{
                var tp = tsneToCanvas(points[ti].x, points[ti].y);
                if (ti === trailStart) ctx.moveTo(tp[0], tp[1]);
                else ctx.lineTo(tp[0], tp[1]);
            }}
            ctx.stroke();

            // Trail dots with fading
            for (var ti = trailStart; ti <= activeIdx; ti++) {{
                var tp = tsneToCanvas(points[ti].x, points[ti].y);
                var alpha = 0.3 + 0.7 * ((ti - trailStart) / Math.max(activeIdx - trailStart, 1));
                ctx.beginPath();
                ctx.arc(tp[0], tp[1], 4, 0, Math.PI * 2);
                ctx.fillStyle = hexToRGBA(color, alpha);
                ctx.fill();
            }}

            // Pulsing active point
            var ap = tsneToCanvas(points[activeIdx].x, points[activeIdx].y);
            ctx.beginPath();
            ctx.arc(ap[0], ap[1], pulseSize + 4, 0, Math.PI * 2);
            ctx.fillStyle = hexToRGBA(color, 0.9);
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 2;
            ctx.stroke();
        }} else {{
            // Dimmer trail for non-active clips
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = hexToRGBA(color, 0.35);
            ctx.beginPath();
            for (var ti = trailStart; ti <= activeIdx; ti++) {{
                var tp = tsneToCanvas(points[ti].x, points[ti].y);
                if (ti === trailStart) ctx.moveTo(tp[0], tp[1]);
                else ctx.lineTo(tp[0], tp[1]);
            }}
            ctx.stroke();

            // Active dot (no pulse)
            var ap = tsneToCanvas(points[activeIdx].x, points[activeIdx].y);
            ctx.beginPath();
            ctx.arc(ap[0], ap[1], 5, 0, Math.PI * 2);
            ctx.fillStyle = hexToRGBA(color, 0.6);
            ctx.fill();
            ctx.strokeStyle = hexToRGBA(color, 0.9);
            ctx.lineWidth = 1;
            ctx.stroke();
        }}
    }}

    // Update info box with active clip details
    var ac = clips[activeClipIdx];
    var ai = findActivePoint(ac.points, currentFrame);
    if (ai >= 0 && ai < ac.points.length) {{
        var pt = ac.points[ai];
        infoBox.innerHTML =
            '<b style="color:' + ac.color + '">Clip ' + ac.clipIdx + '</b> | ' +
            'Frame ' + currentFrame + ' | ' +
            'Point ' + (ai + 1) + '/' + ac.points.length + ' | ' +
            'Codes: [' + pt.codes.join(', ') + '] | ' +
            'Frames ' + pt.sf + '\\u2013' + pt.ef;
    }}
}}

function hexToRGBA(hex, alpha) {{
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}}

// === VIDEO SYNC ===
// Use the first video as the time source; sync others to it.
var masterVideo = videos[0];

var animRunning = false;
function animLoop() {{
    if (masterVideo && !masterVideo.paused) {{
        currentFrame = Math.floor(masterVideo.currentTime * FPS);
        // Sync other videos to master
        for (var i = 1; i < videos.length; i++) {{
            if (videos[i] && Math.abs(videos[i].currentTime - masterVideo.currentTime) > 0.1) {{
                videos[i].currentTime = masterVideo.currentTime;
            }}
        }}
        drawCanvas();
    }}
    if (animRunning) requestAnimationFrame(animLoop);
}}

masterVideo.addEventListener('play', function() {{
    animRunning = true;
    document.getElementById('playPauseBtn').textContent = 'Pause All';
    animLoop();
}});
masterVideo.addEventListener('pause', function() {{
    animRunning = false;
    document.getElementById('playPauseBtn').textContent = 'Play All';
}});
masterVideo.addEventListener('timeupdate', function() {{
    currentFrame = Math.floor(masterVideo.currentTime * FPS);
    drawCanvas();
}});
masterVideo.addEventListener('seeked', function() {{
    // Sync all on seek
    for (var i = 1; i < videos.length; i++) {{
        if (videos[i]) videos[i].currentTime = masterVideo.currentTime;
    }}
    currentFrame = Math.floor(masterVideo.currentTime * FPS);
    drawCanvas();
}});

// === CLIP SELECTION (highlight) ===
function setActiveClip(idx) {{
    activeClipIdx = idx;
    // Update cell borders
    var cells = document.querySelectorAll('.video-cell');
    cells.forEach(function(cell, i) {{
        cell.classList.toggle('active', i === idx);
        if (i === idx) {{
            cell.style.borderColor = clips[idx].color;
        }} else {{
            cell.style.borderColor = 'transparent';
        }}
    }});
    drawCanvas();
}}

// === PLAYBACK CONTROLS ===
function togglePlay() {{
    if (masterVideo.paused) {{
        // Play all
        for (var i = 0; i < videos.length; i++) {{
            if (videos[i]) videos[i].play().catch(function() {{}});
        }}
    }} else {{
        // Pause all
        for (var i = 0; i < videos.length; i++) {{
            if (videos[i]) videos[i].pause();
        }}
    }}
}}

function setSpeed(speed, btn) {{
    for (var i = 0; i < videos.length; i++) {{
        if (videos[i]) videos[i].playbackRate = speed;
    }}
    document.querySelectorAll('.controls-bar button[data-speed]').forEach(function(b) {{
        b.classList.toggle('active', b === btn);
    }});
}}

// === INIT ===
(function init() {{
    // Build legend
    var legend = document.getElementById('legend');
    clips.forEach(function(clip) {{
        var item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML =
            '<span class="legend-dot" style="background:' + clip.color + '"></span>' +
            'Clip ' + clip.clipIdx + ' (mvmt=' + clip.movement.toFixed(3) +
            ', avg ' + clip.avgFrames.toFixed(0) + 'f/trans)';
        legend.appendChild(item);
    }});

    // Set first clip as active
    setActiveClip(0);
    drawCanvas();
}})();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    logging.info(f"Saved t-SNE trajectory HTML to {output_path}")
    return str(output_path)


def generate_tsne_trajectory_html_multicategory(
    category_data_list: list[CategoryData],
    clip_data_by_idx: dict[int, ClipTrajectoryData],
    tsne_coords_per_category: dict[str, np.ndarray],
    clip_point_ranges_per_category: dict[str, dict[int, tuple[int, int]]],
    videos_by_idx: dict[int, str],
    k: int,
    fps: int,
    output_path: str | Path,
    title: str = "t-SNE Skill-Space Trajectory Viewer",
) -> str:
    """Generate multicategory trajectory HTML viewer.

    Creates an HTML page with category buttons at the top. Each category has
    its own t-SNE embedding. Switching categories shows the relevant videos
    and redraws the canvas with that category's t-SNE coordinates.

    Args:
        category_data_list: List of CategoryData for each category.
        clip_data_by_idx: All unique ClipTrajectoryData keyed by clip_idx.
        tsne_coords_per_category: Map from category key to 2D t-SNE coords.
        clip_point_ranges_per_category: Map from category key to
            {clip_idx: (start, end)} index ranges in that category's coords.
        videos_by_idx: Map from clip_idx to base64 video data URL.
        k: Number of transitions per point.
        fps: Video frames per second.
        output_path: Path to save the HTML file.

    Returns:
        Path to the saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-category clip point data and bounds for JS
    categories_js = []
    for cat_data in category_data_list:
        cat_key = cat_data.category.value
        tsne_coords = tsne_coords_per_category[cat_key]
        point_ranges = clip_point_ranges_per_category[cat_key]

        # Per-clip points for this category
        clips_in_cat = {}
        for cd in cat_data.clip_data_list:
            clip_idx = cd.clip_idx
            start, end = point_ranges[clip_idx]
            points = []
            for pi, t in enumerate(cd.transitions):
                coord_idx = start + pi
                x = float(tsne_coords[coord_idx, 0])
                y = float(tsne_coords[coord_idx, 1])
                points.append(
                    {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "sf": t.start_frame,
                        "ef": t.end_frame,
                        "mf": t.midpoint_frame,
                        "codes": list(t.code_sequence),
                    }
                )
            clips_in_cat[clip_idx] = {
                "clipIdx": clip_idx,
                "movement": round(cd.total_movement, 4),
                "avgFrames": round(cd.avg_frames_per_transition, 1),
                "points": points,
            }

        # Compute bounds for this category's t-SNE space
        x_min, x_max = float(tsne_coords[:, 0].min()), float(tsne_coords[:, 0].max())
        y_min, y_max = float(tsne_coords[:, 1].min()), float(tsne_coords[:, 1].max())
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin = 0.08
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin

        clip_idxs = [cd.clip_idx for cd in cat_data.clip_data_list]
        cat_js: dict[str, Any] = {
            "key": cat_key,
            "label": cat_data.label,
            "description": cat_data.description,
            "clipIdxs": clip_idxs,
            "clips": clips_in_cat,
            "bounds": {
                "xMin": round(x_min, 3),
                "xMax": round(x_max, 3),
                "yMin": round(y_min, 3),
                "yMax": round(y_max, 3),
            },
        }

        # Add group coloring if this category has sub-groups
        if cat_data.group_labels is not None:
            group_colors: dict[int, str] = {}
            group_names: dict[int, str] = {}
            for clip_idx, group_name in cat_data.group_labels.items():
                group_colors[clip_idx] = _GROUP_COLORS.get(group_name, "#666")
                group_names[clip_idx] = group_name
            cat_js["groupColors"] = group_colors
            cat_js["groupNames"] = group_names

        categories_js.append(cat_js)

    # Build video HTML for all unique clips (all hidden by default, JS shows active)
    all_unique_clip_idxs = sorted(clip_data_by_idx.keys())
    video_elements_html = ""
    for clip_idx in all_unique_clip_idxs:
        cd = clip_data_by_idx[clip_idx]
        video_src = videos_by_idx.get(clip_idx, "")
        video_elements_html += (
            f'<div class="video-cell" id="vcell-{clip_idx}" '
            f'data-clip-idx="{clip_idx}" '
            f'onclick="setActiveClip({clip_idx})" style="display:none">\n'
            f'  <div class="video-label" id="vlabel-{clip_idx}">'
            f"Clip {clip_idx}"
            f' <span class="movement-tag">'
            f"mvmt={cd.total_movement:.3f}</span></div>\n"
            f'  <video id="vid-{clip_idx}" loop muted playsinline'
            f' src="{video_src}">'
            f"</video>\n"
            f"</div>\n"
        )

    # Build category button bar HTML
    cat_buttons_html = ""
    for i, cat_data in enumerate(category_data_list):
        active_cls = " active" if i == 0 else ""
        cat_buttons_html += (
            f'<button class="cat-btn{active_cls}" '
            f'data-cat-idx="{i}" '
            f'onclick="switchCategory({i})" '
            f'title="{cat_data.description}">'
            f"{cat_data.label}</button>\n"
        )

    # Stats
    total_points = sum(len(cd.transitions) for cd in clip_data_by_idx.values())
    n_unique_clips = len(clip_data_by_idx)

    # Serialize JS data
    categories_json = json.dumps(categories_js)
    colors_json = json.dumps(_CLIP_COLORS)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 16px;
}}
h1 {{
    text-align: center;
    font-size: 1.6em;
    margin-bottom: 12px;
    color: #64b5f6;
}}
.outer-container {{
    max-width: 1400px;
    margin: 0 auto;
}}
.category-bar {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 10px 16px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 12px;
    align-items: center;
}}
.category-bar .label {{
    font-size: 0.85em;
    color: #888;
    margin-right: 4px;
}}
.cat-btn {{
    padding: 6px 16px;
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 14px;
    background: rgba(255,255,255,0.06);
    color: #90caf9;
    cursor: pointer;
    font-size: 0.85em;
    transition: background 0.2s, border-color 0.2s;
}}
.cat-btn:hover {{ background: rgba(100,181,246,0.2); }}
.cat-btn.active {{
    background: rgba(100,181,246,0.3);
    border-color: #64b5f6;
    color: #fff;
}}
.canvas-panel {{
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 16px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
    flex-wrap: wrap;
}}
.canvas-wrapper {{
    flex: 0 0 auto;
}}
.canvas-sidebar {{
    flex: 1;
    min-width: 200px;
}}
canvas {{
    display: block;
    border-radius: 8px;
    background: #ffffff;
    cursor: crosshair;
}}
.video-row {{
    display: flex;
    gap: 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 12px;
    flex-wrap: wrap;
}}
.video-cell {{
    flex: 1;
    min-width: 250px;
    max-width: 400px;
    cursor: pointer;
    border-radius: 10px;
    border: 3px solid transparent;
    padding: 6px;
    transition: border-color 0.2s, background 0.2s;
}}
.video-cell:hover {{
    background: rgba(255,255,255,0.04);
}}
.video-cell.active {{
    border-color: currentColor;
    background: rgba(255,255,255,0.06);
}}
.video-label {{
    font-size: 0.9em;
    font-weight: 600;
    margin-bottom: 6px;
    text-align: center;
}}
.movement-tag {{
    font-size: 0.75em;
    font-weight: 400;
    opacity: 0.7;
}}
.video-cell video {{
    width: 100%;
    border-radius: 8px;
    background: #000;
    display: block;
}}
.controls-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 10px 16px;
    border: 1px solid rgba(100,181,246,0.2);
}}
.controls-bar button {{
    padding: 5px 14px;
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    color: #90caf9;
    cursor: pointer;
    font-size: 0.85em;
}}
.controls-bar button:hover {{ background: rgba(100,181,246,0.2); }}
.controls-bar button.active {{
    background: rgba(100,181,246,0.3);
    border-color: #64b5f6;
    color: #fff;
}}
.stats {{
    font-size: 0.8em;
    color: #888;
    line-height: 1.6;
}}
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
    padding: 8px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.8em;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.info-box {{
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    font-size: 0.8em;
    color: #aaa;
    min-height: 36px;
}}
.cat-desc {{
    margin-top: 8px;
    font-size: 0.8em;
    color: #999;
    font-style: italic;
}}
.spacer {{ flex: 1; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="outer-container">

  <div class="category-bar">
    <span class="label">Category:</span>
    {cat_buttons_html}
  </div>

  <div class="canvas-panel">
    <div class="canvas-wrapper">
      <canvas id="tsneCanvas" width="576" height="576"></canvas>
    </div>
    <div class="canvas-sidebar">
      <div class="legend" id="legend"></div>
      <div class="info-box" id="infoBox">Press Play to begin.</div>
      <div class="cat-desc" id="catDesc"></div>
      <div class="stats">
        {n_unique_clips} unique clips | {total_points} points | k={k}
      </div>
    </div>
  </div>

  <div class="video-row" id="videoRow">
    {video_elements_html}
  </div>

  <div class="controls-bar">
    <button id="playPauseBtn" onclick="togglePlay()">Play All</button>
    <button class="active" data-speed="1" onclick="setSpeed(1, this)">1x</button>
    <button data-speed="0.5" onclick="setSpeed(0.5, this)">0.5x</button>
    <button data-speed="2" onclick="setSpeed(2, this)">2x</button>
    <span class="spacer"></span>
    <span class="stats">Click a video to highlight its trail on the canvas</span>
  </div>

</div>

<script>
// === DATA ===
// Each category has its own clips data, t-SNE coords, and bounds
var categories = {categories_json};
var COLORS = {colors_json};
var FPS = {fps};

// Build a list of all unique clip indices across all categories
var allClipIdxSet = {{}};
categories.forEach(function(cat) {{
    cat.clipIdxs.forEach(function(ci) {{ allClipIdxSet[ci] = true; }});
}});
var allClipIdxs = Object.keys(allClipIdxSet).map(Number).sort(function(a,b){{ return a-b; }});

// === STATE ===
var activeCatIdx = 0;
var activeClipIdx = -1;
var currentFrame = 0;
var trailLength = 12;

// === DOM REFS ===
var canvas = document.getElementById('tsneCanvas');
var ctx = canvas.getContext('2d');
var W = canvas.width, H = canvas.height;
var infoBox = document.getElementById('infoBox');
var catDesc = document.getElementById('catDesc');

// === HELPERS ===
function getCat() {{ return categories[activeCatIdx]; }}
function getActiveCatClipIdxs() {{ return getCat().clipIdxs; }}

function tsneToCanvas(x, y) {{
    var b = getCat().bounds;
    var cx = ((x - b.xMin) / (b.xMax - b.xMin)) * (W - 40) + 20;
    var cy = ((y - b.yMin) / (b.yMax - b.yMin)) * (H - 40) + 20;
    cy = H - cy;
    return [cx, cy];
}}

function getCatClipData(clipIdx) {{
    return getCat().clips[clipIdx];
}}

function getColorForClip(clipIdx) {{
    var cat = getCat();
    if (cat.groupColors && cat.groupColors[clipIdx]) {{
        return cat.groupColors[clipIdx];
    }}
    var catClips = getActiveCatClipIdxs();
    var idx = catClips.indexOf(clipIdx);
    if (idx < 0) return '#666';
    return COLORS[idx % COLORS.length];
}}

function findActivePoint(points, frame) {{
    var bestIdx = -1;
    for (var i = 0; i < points.length; i++) {{
        if (frame >= points[i].sf && frame < points[i].ef) return i;
        if (frame >= points[i].ef) bestIdx = i;
    }}
    if (bestIdx === -1 && points.length > 0) bestIdx = 0;
    return bestIdx;
}}

function hexToRGBA(hex, alpha) {{
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}}

// === CATEGORY MANAGEMENT ===
function switchCategory(catIdx) {{
    // Save current playback time
    var savedTime = 0;
    var wasPlaying = false;
    var oldClips = getActiveCatClipIdxs();
    if (oldClips.length > 0) {{
        var masterVid = document.getElementById('vid-' + oldClips[0]);
        if (masterVid) {{
            savedTime = masterVid.currentTime;
            wasPlaying = !masterVid.paused;
        }}
    }}

    // Pause old videos
    oldClips.forEach(function(ci) {{
        var v = document.getElementById('vid-' + ci);
        if (v) v.pause();
    }});

    activeCatIdx = catIdx;

    // Update category button styles
    document.querySelectorAll('.cat-btn').forEach(function(btn, i) {{
        btn.classList.toggle('active', i === catIdx);
    }});

    // Hide all video cells, show active category's
    allClipIdxs.forEach(function(ci) {{
        document.getElementById('vcell-' + ci).style.display = 'none';
    }});
    var newClips = getActiveCatClipIdxs();
    newClips.forEach(function(ci) {{
        document.getElementById('vcell-' + ci).style.display = '';
        // Update video label color based on group or per-clip color
        var labelEl = document.getElementById('vlabel-' + ci);
        if (labelEl) labelEl.style.color = getColorForClip(ci);
    }});

    // Update category description
    catDesc.textContent = categories[catIdx].description;

    // Rebuild legend
    rebuildLegend();

    // Set active clip to first in new category
    if (newClips.length > 0) {{
        setActiveClip(newClips[0]);

        // Restore playback time
        newClips.forEach(function(ci) {{
            var v = document.getElementById('vid-' + ci);
            if (v) {{
                v.currentTime = savedTime;
                if (wasPlaying) v.play().catch(function(){{}});
            }}
        }});
    }}

    drawCanvas();
}}

function rebuildLegend() {{
    var legend = document.getElementById('legend');
    legend.innerHTML = '';
    var cat = getCat();
    var catClips = getActiveCatClipIdxs();

    if (cat.groupColors) {{
        // Group-based legend: one entry per group
        var seenGroups = {{}};
        catClips.forEach(function(clipIdx) {{
            var groupName = cat.groupNames[clipIdx];
            if (groupName && !seenGroups[groupName]) {{
                seenGroups[groupName] = true;
                var color = cat.groupColors[clipIdx];
                var item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML =
                    '<span class="legend-dot" style="background:' + color + '"></span>' +
                    groupName;
                legend.appendChild(item);
            }}
        }});
        // Also show individual clips below groups
        catClips.forEach(function(clipIdx) {{
            var clip = getCatClipData(clipIdx);
            var color = getColorForClip(clipIdx);
            var groupName = cat.groupNames[clipIdx] || '';
            var item = document.createElement('div');
            item.className = 'legend-item';
            item.style.paddingLeft = '12px';
            item.style.fontSize = '0.75em';
            item.innerHTML =
                '<span class="legend-dot" style="background:' + color + '"></span>' +
                'Clip ' + clip.clipIdx + ' (' + groupName + ', mvmt=' + clip.movement.toFixed(3) + ')';
            legend.appendChild(item);
        }});
    }} else {{
        // Per-clip legend
        catClips.forEach(function(clipIdx, i) {{
            var clip = getCatClipData(clipIdx);
            var color = COLORS[i % COLORS.length];
            var item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML =
                '<span class="legend-dot" style="background:' + color + '"></span>' +
                'Clip ' + clip.clipIdx + ' (mvmt=' + clip.movement.toFixed(3) +
                ', avg ' + clip.avgFrames.toFixed(0) + 'f/trans)';
            legend.appendChild(item);
        }});
    }}
}}

// === DRAWING ===
var pulsePhase = 0;

function drawCanvas() {{
    ctx.clearRect(0, 0, W, H);

    // Draw grid
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 1;
    for (var gx = 0; gx < W; gx += 50) {{
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (var gy = 0; gy < H; gy += 50) {{
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    pulsePhase += 0.08;
    var pulseSize = 3 + Math.sin(pulsePhase) * 2;

    var catClips = getActiveCatClipIdxs();

    // Draw active category clips
    // Non-active-clip first, then active-clip for z-order
    var drawOrder = [];
    catClips.forEach(function(ci) {{
        if (ci !== activeClipIdx) drawOrder.push(ci);
    }});
    if (catClips.indexOf(activeClipIdx) >= 0) drawOrder.push(activeClipIdx);

    for (var di = 0; di < drawOrder.length; di++) {{
        var clipIdx = drawOrder[di];
        var clip = getCatClipData(clipIdx);
        var isActive = (clipIdx === activeClipIdx);
        var color = getColorForClip(clipIdx);
        var points = clip.points;
        var activePointIdx = findActivePoint(points, currentFrame);

        // Draw all points as small dots
        for (var pi = 0; pi < points.length; pi++) {{
            var p = tsneToCanvas(points[pi].x, points[pi].y);
            ctx.beginPath();
            ctx.arc(p[0], p[1], isActive ? 3 : 2, 0, Math.PI * 2);
            ctx.fillStyle = isActive
                ? hexToRGBA(color, 0.4)
                : hexToRGBA(color, 0.15);
            ctx.fill();
        }}

        if (activePointIdx < 0) continue;
        var trailStart = Math.max(0, activePointIdx - trailLength);

        if (isActive) {{
            // Bright trail
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = hexToRGBA(color, 0.7);
            ctx.beginPath();
            for (var ti = trailStart; ti <= activePointIdx; ti++) {{
                var tp = tsneToCanvas(points[ti].x, points[ti].y);
                if (ti === trailStart) ctx.moveTo(tp[0], tp[1]);
                else ctx.lineTo(tp[0], tp[1]);
            }}
            ctx.stroke();

            // Trail dots with fading
            for (var ti = trailStart; ti <= activePointIdx; ti++) {{
                var tp = tsneToCanvas(points[ti].x, points[ti].y);
                var alpha = 0.3 + 0.7 * ((ti - trailStart) / Math.max(activePointIdx - trailStart, 1));
                ctx.beginPath();
                ctx.arc(tp[0], tp[1], 4, 0, Math.PI * 2);
                ctx.fillStyle = hexToRGBA(color, alpha);
                ctx.fill();
            }}

            // Pulsing active point
            var ap = tsneToCanvas(points[activePointIdx].x, points[activePointIdx].y);
            ctx.beginPath();
            ctx.arc(ap[0], ap[1], pulseSize + 4, 0, Math.PI * 2);
            ctx.fillStyle = hexToRGBA(color, 0.9);
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 2;
            ctx.stroke();
        }} else {{
            // Dimmer trail for non-active clips in category
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = hexToRGBA(color, 0.35);
            ctx.beginPath();
            for (var ti = trailStart; ti <= activePointIdx; ti++) {{
                var tp = tsneToCanvas(points[ti].x, points[ti].y);
                if (ti === trailStart) ctx.moveTo(tp[0], tp[1]);
                else ctx.lineTo(tp[0], tp[1]);
            }}
            ctx.stroke();

            // Active dot (no pulse)
            var ap = tsneToCanvas(points[activePointIdx].x, points[activePointIdx].y);
            ctx.beginPath();
            ctx.arc(ap[0], ap[1], 5, 0, Math.PI * 2);
            ctx.fillStyle = hexToRGBA(color, 0.6);
            ctx.fill();
            ctx.strokeStyle = hexToRGBA(color, 0.9);
            ctx.lineWidth = 1;
            ctx.stroke();
        }}
    }}

    // Update info box with active clip details
    var ac = (activeClipIdx >= 0) ? getCatClipData(activeClipIdx) : null;
    if (ac) {{
        var ai = findActivePoint(ac.points, currentFrame);
        if (ai >= 0 && ai < ac.points.length) {{
            var pt = ac.points[ai];
            var color = getColorForClip(activeClipIdx);
            infoBox.innerHTML =
                '<b style="color:' + color + '">Clip ' + ac.clipIdx + '</b> | ' +
                'Frame ' + currentFrame + ' | ' +
                'Point ' + (ai + 1) + '/' + ac.points.length + ' | ' +
                'Codes: [' + pt.codes.join(', ') + '] | ' +
                'Frames ' + pt.sf + '\\u2013' + pt.ef;
        }}
    }}
}}

// === VIDEO SYNC ===
var animRunning = false;

function getMasterVideo() {{
    var catClips = getActiveCatClipIdxs();
    if (catClips.length === 0) return null;
    return document.getElementById('vid-' + catClips[0]);
}}

function animLoop() {{
    var master = getMasterVideo();
    if (master && !master.paused) {{
        currentFrame = Math.floor(master.currentTime * FPS);
        var catClips = getActiveCatClipIdxs();
        for (var i = 1; i < catClips.length; i++) {{
            var v = document.getElementById('vid-' + catClips[i]);
            if (v && Math.abs(v.currentTime - master.currentTime) > 0.1) {{
                v.currentTime = master.currentTime;
            }}
        }}
        drawCanvas();
    }}
    if (animRunning) requestAnimationFrame(animLoop);
}}

// Attach listeners to all videos
allClipIdxs.forEach(function(clipIdx) {{
    var vid = document.getElementById('vid-' + clipIdx);
    if (!vid) return;
    vid.addEventListener('play', function() {{
        if (vid === getMasterVideo()) {{
            animRunning = true;
            document.getElementById('playPauseBtn').textContent = 'Pause All';
            animLoop();
        }}
    }});
    vid.addEventListener('pause', function() {{
        if (vid === getMasterVideo()) {{
            animRunning = false;
            document.getElementById('playPauseBtn').textContent = 'Play All';
        }}
    }});
    vid.addEventListener('timeupdate', function() {{
        if (vid === getMasterVideo()) {{
            currentFrame = Math.floor(vid.currentTime * FPS);
            drawCanvas();
        }}
    }});
    vid.addEventListener('seeked', function() {{
        if (vid === getMasterVideo()) {{
            var catClips = getActiveCatClipIdxs();
            for (var i = 1; i < catClips.length; i++) {{
                var v = document.getElementById('vid-' + catClips[i]);
                if (v) v.currentTime = vid.currentTime;
            }}
            currentFrame = Math.floor(vid.currentTime * FPS);
            drawCanvas();
        }}
    }});
}});

// === CLIP SELECTION (highlight) ===
function setActiveClip(clipIdx) {{
    activeClipIdx = clipIdx;
    var color = getColorForClip(clipIdx);
    allClipIdxs.forEach(function(ci) {{
        var cell = document.getElementById('vcell-' + ci);
        cell.classList.toggle('active', ci === clipIdx);
        cell.style.borderColor = (ci === clipIdx) ? color : 'transparent';
    }});
    drawCanvas();
}}

// === PLAYBACK CONTROLS ===
function togglePlay() {{
    var master = getMasterVideo();
    if (!master) return;
    var catClips = getActiveCatClipIdxs();
    if (master.paused) {{
        catClips.forEach(function(ci) {{
            var v = document.getElementById('vid-' + ci);
            if (v) v.play().catch(function(){{}});
        }});
    }} else {{
        catClips.forEach(function(ci) {{
            var v = document.getElementById('vid-' + ci);
            if (v) v.pause();
        }});
    }}
}}

function setSpeed(speed, btn) {{
    allClipIdxs.forEach(function(ci) {{
        var v = document.getElementById('vid-' + ci);
        if (v) v.playbackRate = speed;
    }});
    document.querySelectorAll('.controls-bar button[data-speed]').forEach(function(b) {{
        b.classList.toggle('active', b === btn);
    }});
}}

// === INIT ===
(function init() {{
    switchCategory(0);
    drawCanvas();
}})();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    logging.info(f"Saved multicategory t-SNE trajectory HTML to {output_path}")
    return str(output_path)


def generate_tsne_static_html(
    category_data_list: list[CategoryData],
    clip_data_by_idx: dict[int, ClipTrajectoryData],
    tsne_coords_per_category: dict[str, np.ndarray],
    clip_point_ranges_per_category: dict[str, dict[int, tuple[int, int]]],
    videos_by_idx: dict[int, str],
    k: int,
    fps: int,
    output_path: str | Path,
    title: str = "t-SNE Skill-Space Static Viewer",
) -> str:
    """Generate static trajectory HTML viewer.

    Same data as the animated viewer but draws all points at once with a
    temporal transparency gradient (opaque=early, transparent=late). No
    animation loop or pulsing dot.

    Args:
        category_data_list: List of CategoryData for each category.
        clip_data_by_idx: All unique ClipTrajectoryData keyed by clip_idx.
        tsne_coords_per_category: Map from category key to 2D t-SNE coords.
        clip_point_ranges_per_category: Map from category key to
            {clip_idx: (start, end)} index ranges in that category's coords.
        videos_by_idx: Map from clip_idx to base64 video data URL.
        k: Number of transitions per point.
        fps: Video frames per second.
        output_path: Path to save the HTML file.

    Returns:
        Path to the saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-category clip point data and bounds for JS (same as animated)
    categories_js = []
    for cat_data in category_data_list:
        cat_key = cat_data.category.value
        tsne_coords = tsne_coords_per_category[cat_key]
        point_ranges = clip_point_ranges_per_category[cat_key]

        clips_in_cat = {}
        for cd in cat_data.clip_data_list:
            clip_idx = cd.clip_idx
            start, end = point_ranges[clip_idx]
            points = []
            for pi, t in enumerate(cd.transitions):
                coord_idx = start + pi
                x = float(tsne_coords[coord_idx, 0])
                y = float(tsne_coords[coord_idx, 1])
                points.append(
                    {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "sf": t.start_frame,
                        "ef": t.end_frame,
                        "mf": t.midpoint_frame,
                        "codes": list(t.code_sequence),
                    }
                )
            clips_in_cat[clip_idx] = {
                "clipIdx": clip_idx,
                "movement": round(cd.total_movement, 4),
                "avgFrames": round(cd.avg_frames_per_transition, 1),
                "points": points,
            }

        x_min, x_max = float(tsne_coords[:, 0].min()), float(tsne_coords[:, 0].max())
        y_min, y_max = float(tsne_coords[:, 1].min()), float(tsne_coords[:, 1].max())
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin = 0.08
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin

        clip_idxs = [cd.clip_idx for cd in cat_data.clip_data_list]
        cat_js: dict[str, Any] = {
            "key": cat_key,
            "label": cat_data.label,
            "description": cat_data.description,
            "clipIdxs": clip_idxs,
            "clips": clips_in_cat,
            "bounds": {
                "xMin": round(x_min, 3),
                "xMax": round(x_max, 3),
                "yMin": round(y_min, 3),
                "yMax": round(y_max, 3),
            },
        }

        if cat_data.group_labels is not None:
            group_colors: dict[int, str] = {}
            group_names: dict[int, str] = {}
            for clip_idx, group_name in cat_data.group_labels.items():
                group_colors[clip_idx] = _GROUP_COLORS.get(group_name, "#666")
                group_names[clip_idx] = group_name
            cat_js["groupColors"] = group_colors
            cat_js["groupNames"] = group_names

        categories_js.append(cat_js)

    # Build video HTML for all unique clips
    all_unique_clip_idxs = sorted(clip_data_by_idx.keys())
    video_elements_html = ""
    for clip_idx in all_unique_clip_idxs:
        cd = clip_data_by_idx[clip_idx]
        video_src = videos_by_idx.get(clip_idx, "")
        video_elements_html += (
            f'<div class="video-cell" id="vcell-{clip_idx}" '
            f'data-clip-idx="{clip_idx}" '
            f'onclick="setActiveClip({clip_idx})" style="display:none">\n'
            f'  <div class="video-label" id="vlabel-{clip_idx}">'
            f"Clip {clip_idx}"
            f' <span class="movement-tag">'
            f"mvmt={cd.total_movement:.3f}</span></div>\n"
            f'  <video id="vid-{clip_idx}" loop muted playsinline'
            f' src="{video_src}">'
            f"</video>\n"
            f"</div>\n"
        )

    # Build category button bar HTML
    cat_buttons_html = ""
    for i, cat_data in enumerate(category_data_list):
        active_cls = " active" if i == 0 else ""
        cat_buttons_html += (
            f'<button class="cat-btn{active_cls}" '
            f'data-cat-idx="{i}" '
            f'onclick="switchCategory({i})" '
            f'title="{cat_data.description}">'
            f"{cat_data.label}</button>\n"
        )

    total_points = sum(len(cd.transitions) for cd in clip_data_by_idx.values())
    n_unique_clips = len(clip_data_by_idx)

    categories_json = json.dumps(categories_js)
    colors_json = json.dumps(_CLIP_COLORS)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 16px;
}}
h1 {{
    text-align: center;
    font-size: 1.6em;
    margin-bottom: 12px;
    color: #64b5f6;
}}
.outer-container {{
    max-width: 1400px;
    margin: 0 auto;
}}
.category-bar {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 10px 16px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 12px;
    align-items: center;
}}
.category-bar .label {{
    font-size: 0.85em;
    color: #888;
    margin-right: 4px;
}}
.cat-btn {{
    padding: 6px 16px;
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 14px;
    background: rgba(255,255,255,0.06);
    color: #90caf9;
    cursor: pointer;
    font-size: 0.85em;
    transition: background 0.2s, border-color 0.2s;
}}
.cat-btn:hover {{ background: rgba(100,181,246,0.2); }}
.cat-btn.active {{
    background: rgba(100,181,246,0.3);
    border-color: #64b5f6;
    color: #fff;
}}
.canvas-panel {{
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 16px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
    flex-wrap: wrap;
}}
.canvas-wrapper {{
    flex: 0 0 auto;
}}
.canvas-sidebar {{
    flex: 1;
    min-width: 200px;
}}
canvas {{
    display: block;
    border-radius: 8px;
    background: #ffffff;
    cursor: crosshair;
}}
.video-row {{
    display: flex;
    gap: 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 12px;
    flex-wrap: wrap;
}}
.video-cell {{
    flex: 1;
    min-width: 250px;
    max-width: 400px;
    cursor: pointer;
    border-radius: 10px;
    border: 3px solid transparent;
    padding: 6px;
    transition: border-color 0.2s, background 0.2s;
}}
.video-cell:hover {{
    background: rgba(255,255,255,0.04);
}}
.video-cell.active {{
    border-color: currentColor;
    background: rgba(255,255,255,0.06);
}}
.video-label {{
    font-size: 0.9em;
    font-weight: 600;
    margin-bottom: 6px;
    text-align: center;
}}
.movement-tag {{
    font-size: 0.75em;
    font-weight: 400;
    opacity: 0.7;
}}
.video-cell video {{
    width: 100%;
    border-radius: 8px;
    background: #000;
    display: block;
}}
.controls-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 10px 16px;
    border: 1px solid rgba(100,181,246,0.2);
}}
.controls-bar button {{
    padding: 5px 14px;
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    color: #90caf9;
    cursor: pointer;
    font-size: 0.85em;
}}
.controls-bar button:hover {{ background: rgba(100,181,246,0.2); }}
.controls-bar button.active {{
    background: rgba(100,181,246,0.3);
    border-color: #64b5f6;
    color: #fff;
}}
.stats {{
    font-size: 0.8em;
    color: #888;
    line-height: 1.6;
}}
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
    padding: 8px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.8em;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.info-box {{
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    font-size: 0.8em;
    color: #aaa;
    min-height: 36px;
}}
.cat-desc {{
    margin-top: 8px;
    font-size: 0.8em;
    color: #999;
    font-style: italic;
}}
.temporal-legend {{
    margin-top: 8px;
    padding: 8px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    font-size: 0.75em;
    color: #aaa;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.temporal-bar {{
    width: 120px;
    height: 10px;
    border-radius: 4px;
    background: linear-gradient(to right, rgba(150,150,150,1.0), rgba(150,150,150,0.15));
    border: 1px solid rgba(255,255,255,0.1);
}}
.spacer {{ flex: 1; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="outer-container">

  <div class="category-bar">
    <span class="label">Category:</span>
    {cat_buttons_html}
  </div>

  <div class="canvas-panel">
    <div class="canvas-wrapper">
      <canvas id="tsneCanvas" width="576" height="576"></canvas>
    </div>
    <div class="canvas-sidebar">
      <div class="legend" id="legend"></div>
      <div class="temporal-legend">
        <span>Early (opaque)</span>
        <div class="temporal-bar"></div>
        <span>Late (transparent)</span>
      </div>
      <div class="info-box" id="infoBox">Click a clip to highlight its trajectory.</div>
      <div class="cat-desc" id="catDesc"></div>
      <div class="stats">
        {n_unique_clips} unique clips | {total_points} points | k={k}
      </div>
    </div>
  </div>

  <div class="video-row" id="videoRow">
    {video_elements_html}
  </div>

  <div class="controls-bar">
    <button id="playPauseBtn" onclick="togglePlay()">Play All</button>
    <button class="active" data-speed="1" onclick="setSpeed(1, this)">1x</button>
    <button data-speed="0.5" onclick="setSpeed(0.5, this)">0.5x</button>
    <button data-speed="2" onclick="setSpeed(2, this)">2x</button>
    <span class="spacer"></span>
    <span class="stats">Click a video to highlight its trajectory</span>
  </div>

</div>

<script>
// === DATA ===
var categories = {categories_json};
var COLORS = {colors_json};
var FPS = {fps};

var allClipIdxSet = {{}};
categories.forEach(function(cat) {{
    cat.clipIdxs.forEach(function(ci) {{ allClipIdxSet[ci] = true; }});
}});
var allClipIdxs = Object.keys(allClipIdxSet).map(Number).sort(function(a,b){{ return a-b; }});

// === STATE ===
var activeCatIdx = 0;
var activeClipIdx = -1;

// === DOM REFS ===
var canvas = document.getElementById('tsneCanvas');
var ctx = canvas.getContext('2d');
var W = canvas.width, H = canvas.height;
var infoBox = document.getElementById('infoBox');
var catDesc = document.getElementById('catDesc');

// === HELPERS ===
function getCat() {{ return categories[activeCatIdx]; }}
function getActiveCatClipIdxs() {{ return getCat().clipIdxs; }}

function tsneToCanvas(x, y) {{
    var b = getCat().bounds;
    var cx = ((x - b.xMin) / (b.xMax - b.xMin)) * (W - 40) + 20;
    var cy = ((y - b.yMin) / (b.yMax - b.yMin)) * (H - 40) + 20;
    cy = H - cy;
    return [cx, cy];
}}

function getCatClipData(clipIdx) {{
    return getCat().clips[clipIdx];
}}

function getColorForClip(clipIdx) {{
    var cat = getCat();
    if (cat.groupColors && cat.groupColors[clipIdx]) {{
        return cat.groupColors[clipIdx];
    }}
    var catClips = getActiveCatClipIdxs();
    var idx = catClips.indexOf(clipIdx);
    if (idx < 0) return '#666';
    return COLORS[idx % COLORS.length];
}}

function hexToRGBA(hex, alpha) {{
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}}

// === DRAWING (STATIC) ===
function drawCanvas() {{
    ctx.clearRect(0, 0, W, H);

    // Draw grid
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 1;
    for (var gx = 0; gx < W; gx += 50) {{
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (var gy = 0; gy < H; gy += 50) {{
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    var catClips = getActiveCatClipIdxs();

    // Draw non-active clips first, then active clip on top
    var drawOrder = [];
    catClips.forEach(function(ci) {{
        if (ci !== activeClipIdx) drawOrder.push(ci);
    }});
    if (catClips.indexOf(activeClipIdx) >= 0) drawOrder.push(activeClipIdx);

    for (var di = 0; di < drawOrder.length; di++) {{
        var clipIdx = drawOrder[di];
        var clip = getCatClipData(clipIdx);
        var isActive = (clipIdx === activeClipIdx);
        var color = getColorForClip(clipIdx);
        var points = clip.points;
        var N = points.length;
        if (N === 0) continue;

        // Draw all points with temporal transparency (no connecting lines)
        for (var pi = 0; pi < N; pi++) {{
            var alpha = 1.0 - (pi / Math.max(N - 1, 1)) * 0.85;
            if (!isActive) alpha *= 0.4;
            var p = tsneToCanvas(points[pi].x, points[pi].y);
            ctx.beginPath();
            ctx.arc(p[0], p[1], isActive ? 4 : 2.5, 0, Math.PI * 2);
            ctx.fillStyle = hexToRGBA(color, alpha);
            ctx.fill();
            if (isActive) {{
                ctx.strokeStyle = hexToRGBA(color, alpha * 0.5);
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }}
        }}

        // Mark start point with a larger circle
        if (isActive) {{
            var sp = tsneToCanvas(points[0].x, points[0].y);
            ctx.beginPath();
            ctx.arc(sp[0], sp[1], 7, 0, Math.PI * 2);
            ctx.fillStyle = hexToRGBA(color, 1.0);
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }}
    }}

    // Update info box
    if (activeClipIdx >= 0) {{
        var ac = getCatClipData(activeClipIdx);
        if (ac) {{
            var color = getColorForClip(activeClipIdx);
            infoBox.innerHTML =
                '<b style="color:' + color + '">Clip ' + ac.clipIdx + '</b> | ' +
                ac.points.length + ' points | mvmt=' + ac.movement.toFixed(3);
        }}
    }}
}}

// === CATEGORY MANAGEMENT ===
function switchCategory(catIdx) {{
    var savedTime = 0;
    var wasPlaying = false;
    var oldClips = getActiveCatClipIdxs();
    if (oldClips.length > 0) {{
        var masterVid = document.getElementById('vid-' + oldClips[0]);
        if (masterVid) {{
            savedTime = masterVid.currentTime;
            wasPlaying = !masterVid.paused;
        }}
    }}

    oldClips.forEach(function(ci) {{
        var v = document.getElementById('vid-' + ci);
        if (v) v.pause();
    }});

    activeCatIdx = catIdx;

    document.querySelectorAll('.cat-btn').forEach(function(btn, i) {{
        btn.classList.toggle('active', i === catIdx);
    }});

    allClipIdxs.forEach(function(ci) {{
        document.getElementById('vcell-' + ci).style.display = 'none';
    }});
    var newClips = getActiveCatClipIdxs();
    newClips.forEach(function(ci) {{
        document.getElementById('vcell-' + ci).style.display = '';
        var labelEl = document.getElementById('vlabel-' + ci);
        if (labelEl) labelEl.style.color = getColorForClip(ci);
    }});

    catDesc.textContent = categories[catIdx].description;
    rebuildLegend();

    if (newClips.length > 0) {{
        setActiveClip(newClips[0]);
        newClips.forEach(function(ci) {{
            var v = document.getElementById('vid-' + ci);
            if (v) {{
                v.currentTime = savedTime;
                if (wasPlaying) v.play().catch(function(){{}});
            }}
        }});
    }}

    drawCanvas();
}}

function rebuildLegend() {{
    var legend = document.getElementById('legend');
    legend.innerHTML = '';
    var cat = getCat();
    var catClips = getActiveCatClipIdxs();

    if (cat.groupColors) {{
        var seenGroups = {{}};
        catClips.forEach(function(clipIdx) {{
            var groupName = cat.groupNames[clipIdx];
            if (groupName && !seenGroups[groupName]) {{
                seenGroups[groupName] = true;
                var color = cat.groupColors[clipIdx];
                var item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML =
                    '<span class="legend-dot" style="background:' + color + '"></span>' +
                    groupName;
                legend.appendChild(item);
            }}
        }});
        catClips.forEach(function(clipIdx) {{
            var clip = getCatClipData(clipIdx);
            var color = getColorForClip(clipIdx);
            var groupName = cat.groupNames[clipIdx] || '';
            var item = document.createElement('div');
            item.className = 'legend-item';
            item.style.paddingLeft = '12px';
            item.style.fontSize = '0.75em';
            item.innerHTML =
                '<span class="legend-dot" style="background:' + color + '"></span>' +
                'Clip ' + clip.clipIdx + ' (' + groupName + ', mvmt=' + clip.movement.toFixed(3) + ')';
            legend.appendChild(item);
        }});
    }} else {{
        catClips.forEach(function(clipIdx, i) {{
            var clip = getCatClipData(clipIdx);
            var color = COLORS[i % COLORS.length];
            var item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML =
                '<span class="legend-dot" style="background:' + color + '"></span>' +
                'Clip ' + clip.clipIdx + ' (mvmt=' + clip.movement.toFixed(3) +
                ', avg ' + clip.avgFrames.toFixed(0) + 'f/trans)';
            legend.appendChild(item);
        }});
    }}
}}

// === CLIP SELECTION ===
function setActiveClip(clipIdx) {{
    activeClipIdx = clipIdx;
    var color = getColorForClip(clipIdx);
    allClipIdxs.forEach(function(ci) {{
        var cell = document.getElementById('vcell-' + ci);
        cell.classList.toggle('active', ci === clipIdx);
        cell.style.borderColor = (ci === clipIdx) ? color : 'transparent';
    }});
    drawCanvas();
}}

// === VIDEO PLAYBACK CONTROLS ===
function getMasterVideo() {{
    var catClips = getActiveCatClipIdxs();
    if (catClips.length === 0) return null;
    return document.getElementById('vid-' + catClips[0]);
}}

function togglePlay() {{
    var master = getMasterVideo();
    if (!master) return;
    var catClips = getActiveCatClipIdxs();
    if (master.paused) {{
        catClips.forEach(function(ci) {{
            var v = document.getElementById('vid-' + ci);
            if (v) v.play().catch(function(){{}});
        }});
        document.getElementById('playPauseBtn').textContent = 'Pause All';
    }} else {{
        catClips.forEach(function(ci) {{
            var v = document.getElementById('vid-' + ci);
            if (v) v.pause();
        }});
        document.getElementById('playPauseBtn').textContent = 'Play All';
    }}
}}

function setSpeed(speed, btn) {{
    allClipIdxs.forEach(function(ci) {{
        var v = document.getElementById('vid-' + ci);
        if (v) v.playbackRate = speed;
    }});
    document.querySelectorAll('.controls-bar button[data-speed]').forEach(function(b) {{
        b.classList.toggle('active', b === btn);
    }});
}}

// Sync videos (without animation loop - just sync on time updates)
allClipIdxs.forEach(function(clipIdx) {{
    var vid = document.getElementById('vid-' + clipIdx);
    if (!vid) return;
    vid.addEventListener('seeked', function() {{
        if (vid === getMasterVideo()) {{
            var catClips = getActiveCatClipIdxs();
            for (var i = 1; i < catClips.length; i++) {{
                var v = document.getElementById('vid-' + catClips[i]);
                if (v) v.currentTime = vid.currentTime;
            }}
        }}
    }});
    vid.addEventListener('timeupdate', function() {{
        if (vid === getMasterVideo()) {{
            var catClips = getActiveCatClipIdxs();
            for (var i = 1; i < catClips.length; i++) {{
                var v = document.getElementById('vid-' + catClips[i]);
                if (v && Math.abs(v.currentTime - vid.currentTime) > 0.1) {{
                    v.currentTime = vid.currentTime;
                }}
            }}
        }}
    }});
}});

// === INIT ===
(function init() {{
    switchCategory(0);
    drawCanvas();
}})();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    logging.info(f"Saved static t-SNE trajectory HTML to {output_path}")
    return str(output_path)


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================


def run_tsne_trajectory_analysis(
    results: Sequence[InferenceResult],
    codebook: np.ndarray,
    output_dir: Path,
    cfg: dict[str, Any] | None = None,
    env: Any | None = None,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> dict[str, Any]:
    """Run the full t-SNE skill-space trajectory analysis.

    Selects clips across 5 movement categories, extracts k-transition features,
    runs t-SNE separately per category, renders videos once per unique clip, and
    generates a multicategory HTML viewer.

    Args:
        results: List of InferenceResult objects.
        codebook: Codebook array of shape [num_codes, latent_dim].
        output_dir: Directory for output files.
        cfg: Configuration dict with keys: n_clips, k_transitions,
            tsne_perplexity, render_videos.
        env: Environment for video rendering (optional).
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.

    Returns:
        Dict with output paths and summary data.
    """
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clips = cfg.get("n_clips", 3)
    k = cfg.get("k_transitions", 8)
    perplexity = cfg.get("tsne_perplexity", 30)
    render_videos = cfg.get("render_videos", True)

    logging.info(f"t-SNE trajectory analysis: n_clips={n_clips}, k={k}")

    xy_vs_xyz_cfg = cfg.get("xy_vs_xyz", {})

    # Step 1: Select clips for all 5 movement categories
    logging.info("  Selecting clips by movement category...")
    categories_map = select_clips_by_category(
        results, n_clips, xy_vs_xyz_cfg=xy_vs_xyz_cfg
    )

    # Step 2: Deduplicate unique clips across all categories
    unique_results: dict[int, InferenceResult] = {}
    for cat_results in categories_map.values():
        for r in cat_results:
            unique_results[r.clip_idx] = r

    if not unique_results:
        logging.warning("  No clips found. Aborting t-SNE trajectory analysis.")
        return {"html_path": None, "summary": {}}

    logging.info(f"  {len(unique_results)} unique clips across all categories")

    # Step 3: Extract k-transitions once per unique clip
    logging.info(f"  Extracting k={k} transitions...")
    clip_data_by_idx: dict[int, ClipTrajectoryData] = {}

    for clip_idx, result in unique_results.items():
        transitions = extract_k_transitions_for_clip(result, k, codebook)
        movement = compute_clip_movement(result)

        if not transitions:
            logging.warning(f"  Clip {clip_idx}: no k-transitions extracted, skipping")
            continue

        avg_frames = np.mean([t.end_frame - t.start_frame for t in transitions])

        clip_data_by_idx[clip_idx] = ClipTrajectoryData(
            clip_idx=clip_idx,
            result=result,
            transitions=transitions,
            total_movement=movement,
            avg_frames_per_transition=float(avg_frames),
        )

        logging.info(
            f"  Clip {clip_idx}: {len(transitions)} points, "
            f"avg {avg_frames:.1f} frames/transition"
        )

    if not clip_data_by_idx:
        logging.warning("  No clips with valid transitions. Aborting.")
        return {"html_path": None, "summary": {}}

    # Step 4: Build CategoryData; gather transitions per category
    category_data_list: list[CategoryData] = []
    # Per-category transitions and ranges (shared between t-SNE and UMAP)
    cat_transitions_map: dict[str, list[KTransition]] = {}
    clip_point_ranges_per_category: dict[str, dict[int, tuple[int, int]]] = {}

    for cat in MovementCategory:
        cat_results = categories_map.get(cat, [])
        cat_clips = [
            clip_data_by_idx[r.clip_idx]
            for r in cat_results
            if r.clip_idx in clip_data_by_idx
        ]
        if not cat_clips:
            logging.info(f"  Skipping category {cat.value}: no valid clips")
            continue

        label, description = _CATEGORY_META[cat]

        # Build group labels for XY_VS_XYZ category
        group_labels = None
        if cat == MovementCategory.XY_VS_XYZ:
            n_per_group = xy_vs_xyz_cfg.get("n_clips_per_group", 3)
            group_labels = {}
            for gi, cd in enumerate(cat_clips):
                group_labels[cd.clip_idx] = (
                    "HIGH_XY" if gi < n_per_group else "HIGH_XYZ"
                )

        category_data_list.append(
            CategoryData(
                category=cat,
                label=label,
                description=description,
                clip_data_list=cat_clips,
                group_labels=group_labels,
            )
        )

        # Gather transitions for this category only
        cat_key = cat.value
        cat_transitions: list[KTransition] = []
        cat_ranges: dict[int, tuple[int, int]] = {}
        for cd in cat_clips:
            start = len(cat_transitions)
            cat_transitions.extend(cd.transitions)
            cat_ranges[cd.clip_idx] = (start, len(cat_transitions))

        cat_transitions_map[cat_key] = cat_transitions
        clip_point_ranges_per_category[cat_key] = cat_ranges

    if not category_data_list:
        logging.warning("  No categories with valid clips. Aborting.")
        return {"html_path": None, "summary": {}}

    # Run t-SNE per category
    tsne_coords_per_category: dict[str, np.ndarray] = {}
    for cat_key, cat_transitions in cat_transitions_map.items():
        logging.info(f"  Running t-SNE for {cat_key}: {len(cat_transitions)} points")
        coords = compute_tsne_embedding(cat_transitions, perplexity)
        if coords is not None:
            tsne_coords_per_category[cat_key] = coords

    # Run UMAP per category
    umap_n_neighbors = cfg.get("umap_n_neighbors", 15)
    umap_min_dist = cfg.get("umap_min_dist", 0.1)
    umap_coords_per_category: dict[str, np.ndarray] = {}
    for cat_key, cat_transitions in cat_transitions_map.items():
        logging.info(f"  Running UMAP for {cat_key}: {len(cat_transitions)} points")
        coords = compute_umap_embedding(
            cat_transitions,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
        )
        if coords is not None:
            umap_coords_per_category[cat_key] = coords

    # Step 5: Render videos once per unique clip
    videos_by_idx: dict[int, str] = {}
    if render_videos and env is not None:
        logging.info("  Rendering clip videos...")
        for clip_idx in sorted(clip_data_by_idx.keys()):
            cd = clip_data_by_idx[clip_idx]
            logging.info(f"    Rendering clip {clip_idx}...")
            b64 = render_clip_video_for_tsne(
                env=env,
                result=cd.result,
                camera=camera,
                width=width,
                height=height,
                fps=fps,
            )
            videos_by_idx[clip_idx] = b64
        logging.info(f"  Rendered {len(videos_by_idx)} unique videos")
    else:
        videos_by_idx = {idx: "" for idx in clip_data_by_idx}

    # Step 6: Generate HTML viewers for both t-SNE and UMAP (animated + static)
    # Filter category lists to only those with valid embeddings
    tsne_cats = [
        cd for cd in category_data_list if cd.category.value in tsne_coords_per_category
    ]
    umap_cats = [
        cd for cd in category_data_list if cd.category.value in umap_coords_per_category
    ]

    html_path = output_dir / "tsne_trajectory.html"
    generate_tsne_trajectory_html_multicategory(
        category_data_list=tsne_cats,
        clip_data_by_idx=clip_data_by_idx,
        tsne_coords_per_category=tsne_coords_per_category,
        clip_point_ranges_per_category=clip_point_ranges_per_category,
        videos_by_idx=videos_by_idx,
        k=k,
        fps=fps,
        output_path=html_path,
    )

    static_html_path = output_dir / "tsne_trajectory_static.html"
    generate_tsne_static_html(
        category_data_list=tsne_cats,
        clip_data_by_idx=clip_data_by_idx,
        tsne_coords_per_category=tsne_coords_per_category,
        clip_point_ranges_per_category=clip_point_ranges_per_category,
        videos_by_idx=videos_by_idx,
        k=k,
        fps=fps,
        output_path=static_html_path,
    )

    # UMAP variants
    umap_html_path = output_dir / "umap_trajectory.html"
    generate_tsne_trajectory_html_multicategory(
        category_data_list=umap_cats,
        clip_data_by_idx=clip_data_by_idx,
        tsne_coords_per_category=umap_coords_per_category,
        clip_point_ranges_per_category=clip_point_ranges_per_category,
        videos_by_idx=videos_by_idx,
        k=k,
        fps=fps,
        output_path=umap_html_path,
        title="UMAP Skill-Space Trajectory Viewer",
    )

    umap_static_html_path = output_dir / "umap_trajectory_static.html"
    generate_tsne_static_html(
        category_data_list=umap_cats,
        clip_data_by_idx=clip_data_by_idx,
        tsne_coords_per_category=umap_coords_per_category,
        clip_point_ranges_per_category=clip_point_ranges_per_category,
        videos_by_idx=videos_by_idx,
        k=k,
        fps=fps,
        output_path=umap_static_html_path,
        title="UMAP Skill-Space Static Viewer",
    )

    # Save JSON summary
    total_points = sum(len(cd.transitions) for cd in clip_data_by_idx.values())
    summary = {
        "n_unique_clips": len(clip_data_by_idx),
        "k_transitions": k,
        "total_points": total_points,
        "tsne_perplexity": perplexity,
        "categories": [
            {
                "category": cat_data.category.value,
                "label": cat_data.label,
                "n_clips": len(cat_data.clip_data_list),
                "clip_idxs": [cd.clip_idx for cd in cat_data.clip_data_list],
            }
            for cat_data in category_data_list
        ],
        "clips": [
            {
                "clip_idx": cd.clip_idx,
                "total_movement": cd.total_movement,
                "n_transitions": len(cd.transitions),
                "avg_frames_per_transition": cd.avg_frames_per_transition,
            }
            for cd in clip_data_by_idx.values()
        ],
    }

    json_path = output_dir / "tsne_trajectory_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"  t-SNE trajectory analysis complete: {len(clip_data_by_idx)} unique clips, "
        f"{len(category_data_list)} categories, {total_points} points"
    )

    return {
        "html_path": str(html_path),
        "static_html_path": str(static_html_path),
        "umap_html_path": str(umap_html_path),
        "umap_static_html_path": str(umap_static_html_path),
        "json_path": str(json_path),
        "summary": summary,
    }
