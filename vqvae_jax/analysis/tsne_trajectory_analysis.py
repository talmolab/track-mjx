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
) -> np.ndarray:
    """Run t-SNE on all k-transition embeddings.

    Args:
        all_transitions: Combined list of KTransition from all clips.
        perplexity: t-SNE perplexity parameter.

    Returns:
        2D coordinates of shape [N_total, 2].
    """
    from sklearn.manifold import TSNE

    embeddings = np.stack([t.embedding for t in all_transitions])
    n_samples = len(embeddings)

    # Adjust perplexity if too large for the number of samples
    effective_perplexity = min(perplexity, max(5.0, (n_samples - 1) / 3.0))
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
    background: #0d1117;
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
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
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
            ctx.strokeStyle = '#fff';
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

    Selects high-movement clips, extracts k-transition features, runs t-SNE,
    renders videos, and generates a synchronized HTML viewer.

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

    # Step 1: Select high-movement clips
    logging.info("  Selecting high-movement clips...")
    selected_results = select_clips_by_movement(results, n_clips)

    if not selected_results:
        logging.warning("  No clips found. Aborting t-SNE trajectory analysis.")
        return {"html_path": None, "summary": {}}

    # Step 2: Extract k-transitions for each clip
    logging.info(f"  Extracting k={k} transitions...")
    clip_data_list: list[ClipTrajectoryData] = []

    for result in selected_results:
        transitions = extract_k_transitions_for_clip(result, k, codebook)
        movement = compute_clip_movement(result)

        if not transitions:
            logging.warning(
                f"  Clip {result.clip_idx}: no k-transitions extracted, skipping"
            )
            continue

        avg_frames = np.mean([t.end_frame - t.start_frame for t in transitions])

        clip_data_list.append(
            ClipTrajectoryData(
                clip_idx=result.clip_idx,
                result=result,
                transitions=transitions,
                total_movement=movement,
                avg_frames_per_transition=float(avg_frames),
            )
        )

        logging.info(
            f"  Clip {result.clip_idx}: {len(transitions)} points, "
            f"avg {avg_frames:.1f} frames/transition"
        )

    if not clip_data_list:
        logging.warning("  No clips with valid transitions. Aborting.")
        return {"html_path": None, "summary": {}}

    # Step 3: Run t-SNE on all transitions
    all_transitions = []
    for cd in clip_data_list:
        all_transitions.extend(cd.transitions)

    tsne_coords = compute_tsne_embedding(all_transitions, perplexity)

    # Step 4: Render videos if enabled
    videos_b64: list[str] = []
    if render_videos and env is not None:
        logging.info("  Rendering clip videos...")
        for cd in clip_data_list:
            logging.info(f"    Rendering clip {cd.clip_idx}...")
            b64 = render_clip_video_for_tsne(
                env=env,
                result=cd.result,
                camera=camera,
                width=width,
                height=height,
                fps=fps,
            )
            videos_b64.append(b64)
        logging.info(f"  Rendered {len(videos_b64)} videos")
    else:
        # No video rendering - use empty placeholders
        videos_b64 = ["" for _ in clip_data_list]

    # Step 5: Generate HTML viewer
    html_path = output_dir / "tsne_trajectory.html"
    generate_tsne_trajectory_html(
        clip_data_list=clip_data_list,
        tsne_coords=tsne_coords,
        videos_b64=videos_b64,
        k=k,
        fps=fps,
        output_path=html_path,
    )

    # Save JSON summary
    total_points = sum(len(cd.transitions) for cd in clip_data_list)
    summary = {
        "n_clips": len(clip_data_list),
        "k_transitions": k,
        "total_points": total_points,
        "tsne_perplexity": perplexity,
        "clips": [
            {
                "clip_idx": cd.clip_idx,
                "total_movement": cd.total_movement,
                "n_transitions": len(cd.transitions),
                "avg_frames_per_transition": cd.avg_frames_per_transition,
            }
            for cd in clip_data_list
        ],
    }

    json_path = output_dir / "tsne_trajectory_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"  t-SNE trajectory analysis complete: {len(clip_data_list)} clips, "
        f"{total_points} points"
    )

    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
        "summary": summary,
    }
