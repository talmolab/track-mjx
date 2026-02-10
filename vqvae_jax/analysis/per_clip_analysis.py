"""Per-clip analysis for VQ-VAE codes.

This module provides functions for analyzing VQ-VAE code patterns on a per-clip
basis, including transition matrices, community detection, transition graphs,
and generating an interactive HTML visualization with sliders to compare clips.
"""

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult
from .rendering import get_nature_colormap

if TYPE_CHECKING:
    pass


@dataclass
class PerClipAnalysis:
    """Analysis results for a single clip.

    Attributes:
        clip_idx: Index of the clip.
        num_frames: Total frames in the clip.
        code_indices: Array of code indices per frame.
        unique_codes: List of unique codes used in this clip.
        trans_counts: Transition count matrix for this clip.
        trans_probs: Transition probability matrix for this clip.
        communities: Dict mapping code_idx to community_id.
        n_communities: Number of communities detected.
        modularity: Modularity score of community detection.
        code_frame_counts: Dict mapping code_idx to frame count in this clip.
    """

    clip_idx: int
    num_frames: int
    code_indices: np.ndarray
    unique_codes: list[int]
    trans_counts: np.ndarray
    trans_probs: np.ndarray
    communities: dict[int, int]
    n_communities: int
    modularity: float
    code_frame_counts: dict[int, int]


def compute_clip_transition_matrix(
    code_indices: np.ndarray,
    num_codes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition matrix for a single clip."""
    trans_counts = np.zeros((num_codes, num_codes), dtype=np.int32)

    if len(code_indices) < 2:
        return trans_counts, np.zeros_like(trans_counts, dtype=np.float64)

    for i in range(len(code_indices) - 1):
        from_code = int(code_indices[i])
        to_code = int(code_indices[i + 1])
        if 0 <= from_code < num_codes and 0 <= to_code < num_codes:
            trans_counts[from_code, to_code] += 1

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.where(row_sums > 0, trans_counts / row_sums, 0.0)

    return trans_counts, trans_probs


def detect_clip_communities(
    trans_counts: np.ndarray,
    code_indices: np.ndarray,
    n_communities: int | None = None,
    min_codes_for_clustering: int = 3,
) -> tuple[dict[int, int], int, float]:
    """Detect communities in a single clip's transition graph."""
    unique_codes = list(set(int(c) for c in code_indices))
    n_unique = len(unique_codes)

    if n_unique < min_codes_for_clustering:
        logging.debug(f"Only {n_unique} unique codes, skipping clustering")
        return {code: 0 for code in unique_codes}, 1, 0.0

    sub_matrix = np.zeros((n_unique, n_unique), dtype=np.float64)
    for i, code_i in enumerate(unique_codes):
        for j, code_j in enumerate(unique_codes):
            sub_matrix[i, j] = trans_counts[code_i, code_j]

    sym_matrix = sub_matrix + sub_matrix.T

    if n_communities is None:
        n_communities = max(2, min(int(np.sqrt(n_unique)), 8))
    n_communities = min(n_communities, n_unique)

    try:
        from sklearn.cluster import SpectralClustering

        affinity = sym_matrix + 1e-6
        clustering = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = clustering.fit_predict(affinity)

        code_to_community = {unique_codes[i]: int(labels[i]) for i in range(n_unique)}
        modularity = _compute_modularity(sym_matrix, labels)

    except Exception as e:
        logging.warning(f"Spectral clustering failed: {e}")
        code_to_community = {code: 0 for code in unique_codes}
        n_communities = 1
        modularity = 0.0

    return code_to_community, n_communities, modularity


def _compute_modularity(adj_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute modularity of a clustering."""
    m = adj_matrix.sum() / 2
    if m == 0:
        return 0.0

    n = len(labels)
    k = adj_matrix.sum(axis=1)

    Q = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += adj_matrix[i, j] - (k[i] * k[j]) / (2 * m)

    return Q / (2 * m)


def analyze_single_clip(
    result: InferenceResult,
    num_codes: int,
    n_communities: int | None = None,
) -> PerClipAnalysis:
    """Run complete analysis on a single clip."""
    code_indices = result.code_indices
    unique_codes = list(set(int(c) for c in code_indices))

    trans_counts, trans_probs = compute_clip_transition_matrix(code_indices, num_codes)
    communities, n_comms, modularity = detect_clip_communities(
        trans_counts, code_indices, n_communities
    )

    code_frame_counts: dict[int, int] = {}
    for code in code_indices:
        code = int(code)
        code_frame_counts[code] = code_frame_counts.get(code, 0) + 1

    return PerClipAnalysis(
        clip_idx=result.clip_idx,
        num_frames=len(code_indices),
        code_indices=code_indices,
        unique_codes=unique_codes,
        trans_counts=trans_counts,
        trans_probs=trans_probs,
        communities=communities,
        n_communities=n_comms,
        modularity=modularity,
        code_frame_counts=code_frame_counts,
    )


def plot_clip_analysis(
    analysis: PerClipAnalysis,
    num_codes: int,
    figsize: tuple[int, int] = (18, 14),
) -> plt.Figure:
    """Create comprehensive multi-panel visualization for a single clip."""
    fig = plt.figure(figsize=figsize)

    # Create grid: 3 rows, 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    code_colors = get_nature_colormap(num_codes)
    active_codes = sorted(analysis.unique_codes)
    n_active = len(active_codes)

    # Community colors - use distinct, saturated colors
    DISTINCT_COLORS = [
        (0.122, 0.467, 0.706, 1.0),  # Blue
        (1.000, 0.498, 0.055, 1.0),  # Orange
        (0.173, 0.627, 0.173, 1.0),  # Green
        (0.839, 0.153, 0.157, 1.0),  # Red
        (0.580, 0.404, 0.741, 1.0),  # Purple
        (0.549, 0.337, 0.294, 1.0),  # Brown
        (0.890, 0.467, 0.761, 1.0),  # Pink
        (0.498, 0.498, 0.498, 1.0),  # Gray
        (0.737, 0.741, 0.133, 1.0),  # Yellow-green
        (0.090, 0.745, 0.812, 1.0),  # Cyan
    ]
    comm_colors = {
        i: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        for i in range(max(analysis.n_communities, 1))
    }

    # === Row 1, Col 0: Code usage histogram ===
    ax = fig.add_subplot(gs[0, 0])
    codes = sorted(analysis.code_frame_counts.keys())
    counts = [analysis.code_frame_counts[c] for c in codes]
    colors = [code_colors[c] / 255.0 for c in codes]

    ax.bar(range(len(codes)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=45)
    ax.set_xlabel("Code Index", fontsize=9)
    ax.set_ylabel("Frame Count", fontsize=9)
    ax.set_title("Code Usage Distribution", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # === Row 1, Col 1: Transition matrix heatmap ===
    ax = fig.add_subplot(gs[0, 1])

    if n_active > 0:
        sub_probs = np.zeros((n_active, n_active))
        for i, ci in enumerate(active_codes):
            for j, cj in enumerate(active_codes):
                sub_probs[i, j] = analysis.trans_probs[ci, cj]

        im = ax.imshow(sub_probs, cmap="YlOrRd", aspect="equal", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Probability", fontsize=8)

        ax.set_xticks(range(n_active))
        ax.set_yticks(range(n_active))
        ax.set_xticklabels(active_codes, fontsize=6, rotation=45)
        ax.set_yticklabels(active_codes, fontsize=6)
        ax.set_xlabel("To Code", fontsize=9)
        ax.set_ylabel("From Code", fontsize=9)

    ax.set_title(
        f"Transition Matrix ({n_active} codes)", fontsize=10, fontweight="bold"
    )

    # === Row 1, Col 2: Transition graph ===
    ax = fig.add_subplot(gs[0, 2])
    _plot_transition_graph(ax, analysis, code_colors, comm_colors)

    # === Row 2, Col 0-1: Code sequence timeline (dual: code colors + community colors) ===
    ax = fig.add_subplot(gs[1, :2])

    n_frames = len(analysis.code_indices)

    # Create dual timeline: top half = code colors, bottom half = community colors
    timeline_height = 80
    timeline = np.ones((timeline_height, n_frames, 3), dtype=np.uint8) * 255

    # Top section: Code colors
    for i, code in enumerate(analysis.code_indices):
        color = code_colors[int(code)]
        timeline[: timeline_height // 2 - 2, i] = color

    # Separator line
    timeline[timeline_height // 2 - 2 : timeline_height // 2, :] = [50, 50, 50]

    # Bottom section: Community colors
    for i, code in enumerate(analysis.code_indices):
        comm_id = analysis.communities.get(int(code), 0)
        # Convert matplotlib color to RGB uint8
        comm_rgba = comm_colors.get(comm_id, (0.5, 0.5, 0.5, 1.0))
        comm_rgb = [int(c * 255) for c in comm_rgba[:3]]
        timeline[timeline_height // 2 :, i] = comm_rgb

    ax.imshow(timeline, aspect="auto")
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("")
    ax.set_yticks([timeline_height // 4, 3 * timeline_height // 4])
    ax.set_yticklabels(["Codes", "Communities"], fontsize=8)
    ax.set_title(
        f"Code & Community Timeline ({n_frames} frames)", fontsize=10, fontweight="bold"
    )

    # Add frame markers
    for x in range(0, n_frames, 50):
        ax.axvline(x, color="white", alpha=0.3, linewidth=0.5)
        ax.text(x, -3, str(x), fontsize=6, ha="center", color="gray")

    # === Row 2, Col 2: Community assignment ===
    ax = fig.add_subplot(gs[1, 2])

    if analysis.n_communities > 0 and len(analysis.communities) > 0:
        codes_sorted = sorted(analysis.communities.keys())
        comm_labels = [analysis.communities[c] for c in codes_sorted]
        bar_colors = [comm_colors[l] for l in comm_labels]

        ax.bar(
            range(len(codes_sorted)),
            [1] * len(codes_sorted),
            color=bar_colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(codes_sorted)))
        ax.set_xticklabels(codes_sorted, fontsize=7, rotation=45)
        ax.set_xlabel("Code Index", fontsize=9)
        ax.set_yticks([])

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=comm_colors[i])
            for i in range(analysis.n_communities)
        ]
        ax.legend(
            handles,
            [f"C{i}" for i in range(analysis.n_communities)],
            loc="upper right",
            fontsize=7,
            ncol=2,
        )

    ax.set_title(
        f"Community Assignment (n={analysis.n_communities}, Q={analysis.modularity:.3f})",
        fontsize=10,
        fontweight="bold",
    )

    # === Row 3, Col 0: Self-loop probabilities ===
    ax = fig.add_subplot(gs[2, 0])

    self_probs = []
    for code in active_codes:
        self_probs.append(analysis.trans_probs[code, code])

    colors_self = []
    for i, p in enumerate(self_probs):
        if p > 0.7:
            colors_self.append("#E74C3C")
        elif p > 0.5:
            colors_self.append("#F39C12")
        elif p > 0.3:
            colors_self.append("#27AE60")
        else:
            colors_self.append("#3498DB")

    ax.bar(range(len(active_codes)), self_probs, color=colors_self, edgecolor="white")
    ax.set_xticks(range(len(active_codes)))
    ax.set_xticklabels(active_codes, fontsize=7, rotation=45)
    ax.set_xlabel("Code Index", fontsize=9)
    ax.set_ylabel("Self-loop Probability", fontsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Self-loop Probabilities", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # === Row 3, Col 1: Top transitions ===
    ax = fig.add_subplot(gs[2, 1])

    # Find top transitions (excluding self-loops)
    transitions = []
    for i in active_codes:
        for j in active_codes:
            if i != j and analysis.trans_probs[i, j] > 0.01:
                transitions.append((i, j, analysis.trans_probs[i, j]))

    transitions.sort(key=lambda x: x[2], reverse=True)
    top_trans = transitions[:10]

    if top_trans:
        labels = [f"{t[0]}→{t[1]}" for t in top_trans]
        probs = [t[2] for t in top_trans]

        y_pos = range(len(top_trans))
        ax.barh(y_pos, probs, color="#5DADE2", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Probability", fontsize=9)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

    ax.set_title("Top 10 Transitions", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # === Row 3, Col 2: Statistics summary ===
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    # Calculate additional stats
    total_transitions = int(analysis.trans_counts.sum())
    self_loop_total = sum(analysis.trans_counts[c, c] for c in active_codes)
    self_loop_pct = 100 * self_loop_total / max(total_transitions, 1)

    sorted_codes = sorted(
        analysis.code_frame_counts.items(), key=lambda x: x[1], reverse=True
    )

    stats_lines = [
        f"Clip Index:       {analysis.clip_idx}",
        f"Total Frames:     {analysis.num_frames}",
        f"Unique Codes:     {len(analysis.unique_codes)}",
        f"Communities:      {analysis.n_communities}",
        f"Modularity:       {analysis.modularity:.4f}",
        f"Total Transitions:{total_transitions}",
        f"Self-loops:       {self_loop_pct:.1f}%",
        "",
        "Top 5 Codes:",
    ]

    for code, count in sorted_codes[:5]:
        pct = 100 * count / analysis.num_frames
        stats_lines.append(f"  Code {code:2d}: {count:4d} ({pct:5.1f}%)")

    ax.text(
        0.05,
        0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"Clip {analysis.clip_idx} Analysis", fontsize=14, fontweight="bold", y=0.98
    )

    return fig


def _plot_transition_graph(
    ax: plt.Axes,
    analysis: PerClipAnalysis,
    code_colors: np.ndarray,
    comm_colors: dict[int, tuple],
    min_edge_prob: float = 0.05,
) -> None:
    """Plot transition graph on given axes."""
    try:
        import networkx as nx
    except ImportError:
        ax.text(
            0.5,
            0.5,
            "NetworkX not installed",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Transition Graph (unavailable)")
        ax.axis("off")
        return

    active_codes = sorted(analysis.unique_codes)
    n_active = len(active_codes)

    if n_active < 2:
        ax.text(
            0.5,
            0.5,
            "Not enough codes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Transition Graph")
        ax.axis("off")
        return

    G = nx.DiGraph()

    for code in active_codes:
        G.add_node(code)

    edges = []
    for i in active_codes:
        for j in active_codes:
            prob = analysis.trans_probs[i, j]
            if prob >= min_edge_prob and i != j:
                G.add_edge(i, j, weight=prob)
                edges.append((i, j, prob))

    if len(edges) == 0:
        ax.text(
            0.5,
            0.5,
            "No transitions above threshold",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Transition Graph")
        ax.axis("off")
        return

    # Layout
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Node sizes based on frame count
    max_count = max(analysis.code_frame_counts.values())
    node_sizes = []
    for node in G.nodes():
        count = analysis.code_frame_counts.get(node, 1)
        size = 200 + 800 * (count / max_count)
        node_sizes.append(size)

    # Node colors by community
    node_colors = []
    for node in G.nodes():
        comm = analysis.communities.get(node, 0)
        node_colors.append(comm_colors[comm])

    # Draw edges
    edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.4,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=1.5,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=7,
        font_weight="bold",
    )

    ax.set_title("Transition Graph", fontsize=10, fontweight="bold")
    ax.axis("off")


def figure_to_base64(fig: plt.Figure, fmt: str = "png", dpi: int = 100) -> str:
    """Convert matplotlib figure to base64-encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64


def generate_interactive_html(
    analyses: list[PerClipAnalysis],
    num_codes: int,
    output_path: str | Path,
    title: str = "Per-Clip VQ-VAE Analysis",
) -> str:
    """Generate interactive HTML with slider to browse per-clip analyses."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Generating visualizations for {len(analyses)} clips...")
    images_data = []

    for i, analysis in enumerate(analyses):
        logging.info(f"  Processing clip {analysis.clip_idx} ({i+1}/{len(analyses)})")
        fig = plot_clip_analysis(analysis, num_codes)
        img_b64 = figure_to_base64(fig, dpi=100)
        plt.close(fig)

        images_data.append(
            {
                "clip_idx": analysis.clip_idx,
                "image": img_b64,
                "stats": {
                    "num_frames": analysis.num_frames,
                    "unique_codes": len(analysis.unique_codes),
                    "n_communities": analysis.n_communities,
                    "modularity": round(analysis.modularity, 4),
                    "top_code": (
                        max(analysis.code_frame_counts.items(), key=lambda x: x[1])[0]
                        if analysis.code_frame_counts
                        else 0
                    ),
                },
            }
        )

    html_content = _generate_html_template(images_data, title)

    with open(output_path, "w") as f:
        f.write(html_content)

    logging.info(f"Saved interactive HTML to {output_path}")
    return str(output_path)


def _generate_html_template(images_data: list[dict], title: str) -> str:
    """Generate HTML template with embedded JavaScript."""
    js_data = json.dumps(images_data)

    html = f"""<!DOCTYPE html>
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
        .clip-counter {{
            font-size: 18px;
            font-weight: 600;
            color: #4fc3f7;
            min-width: 100px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="controls">
            <div class="slider-row">
                <button class="nav-btn" id="prevBtn" onclick="prevClip()">&#9664; Prev</button>
                <span class="clip-counter" id="clipCounter">1 / {len(images_data)}</span>
                <button class="nav-btn" id="nextBtn" onclick="nextClip()">Next &#9654;</button>
                <div class="slider-wrapper">
                    <input type="range" id="clipSlider" min="0" max="{len(images_data) - 1}" value="0" oninput="updateClip(this.value)">
                </div>
            </div>
            <div class="stats-row" id="statsRow"></div>
            <div class="hint">Use ← → arrow keys or slider to navigate</div>
        </div>
        <div class="image-container">
            <img id="clipImage" src="" alt="Clip Analysis">
        </div>
    </div>
    <script>
        const clipsData = {js_data};
        let idx = 0;
        function updateClip(i) {{
            idx = parseInt(i);
            const c = clipsData[idx];
            document.getElementById('clipImage').src = 'data:image/png;base64,' + c.image;
            document.getElementById('clipCounter').textContent = (idx + 1) + ' / ' + clipsData.length;
            document.getElementById('clipSlider').value = idx;
            document.getElementById('statsRow').innerHTML = `
                <div class="stat-badge"><strong>Clip:</strong> ${{c.clip_idx}}</div>
                <div class="stat-badge"><strong>Frames:</strong> ${{c.stats.num_frames}}</div>
                <div class="stat-badge"><strong>Codes:</strong> ${{c.stats.unique_codes}}</div>
                <div class="stat-badge"><strong>Communities:</strong> ${{c.stats.n_communities}}</div>
                <div class="stat-badge"><strong>Modularity:</strong> ${{c.stats.modularity.toFixed(3)}}</div>
                <div class="stat-badge"><strong>Top Code:</strong> ${{c.stats.top_code}}</div>
            `;
            document.getElementById('prevBtn').disabled = idx === 0;
            document.getElementById('nextBtn').disabled = idx === clipsData.length - 1;
        }}
        function nextClip() {{ if (idx < clipsData.length - 1) updateClip(idx + 1); }}
        function prevClip() {{ if (idx > 0) updateClip(idx - 1); }}
        document.addEventListener('keydown', e => {{
            if (e.key === 'ArrowRight') nextClip();
            else if (e.key === 'ArrowLeft') prevClip();
        }});
        updateClip(0);
    </script>
</body>
</html>"""

    return html


def render_clip_video(
    env: Any,
    result: "InferenceResult",
    analysis: PerClipAnalysis,
    output_path: Path,
    num_codes: int,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    bar_height: int = 50,
) -> str:
    """Render a single clip video with dual code/community timeline bars.

    Args:
        env: Environment with mj_model attribute for rendering.
        result: InferenceResult with qpos data.
        analysis: PerClipAnalysis with community assignments.
        output_path: Path to save the video.
        num_codes: Total number of codes for colormap.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        bar_height: Height of each timeline bar (code and community).

    Returns:
        Path to the rendered video.
    """
    import imageio
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if result.qpos is None or len(result.qpos) == 0:
        logging.warning(f"No qpos data for clip {analysis.clip_idx}")
        return ""

    # Get colormaps
    code_colors = get_nature_colormap(num_codes)

    # Distinct community colors
    DISTINCT_COLORS = [
        [31, 119, 180],  # Blue
        [255, 127, 14],  # Orange
        [44, 160, 44],  # Green
        [214, 39, 40],  # Red
        [148, 103, 189],  # Purple
        [140, 86, 75],  # Brown
        [227, 119, 194],  # Pink
        [127, 127, 127],  # Gray
        [188, 189, 34],  # Yellow-green
        [23, 190, 207],  # Cyan
    ]
    comm_colors = np.array(
        [
            DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
            for i in range(max(analysis.n_communities, 1))
        ],
        dtype=np.uint8,
    )

    # Setup MuJoCo renderer
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    render_height = height - 2 * bar_height
    renderer = mujoco.Renderer(mj_model, height=render_height, width=width)

    # Get camera ID
    cam_id = -1
    if camera:
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        except Exception:
            logging.warning(f"Camera '{camera}' not found, using default")

    n_frames = len(result.qpos)
    indices = analysis.code_indices

    frames = []
    for i in range(n_frames):
        # Set qpos and forward kinematics
        mj_data.qpos[:] = result.qpos[i]
        mujoco.mj_forward(mj_model, mj_data)

        # Render frame
        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)
        render_frame = renderer.render()

        # Create full frame with bars
        full_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        full_frame[:render_height, :] = render_frame

        # === Top bar: Code timeline ===
        code_bar_y = render_height
        for j in range(n_frames):
            x_start = int(j * width / n_frames)
            x_end = int((j + 1) * width / n_frames)
            code_idx = int(indices[j]) if j < len(indices) else 0
            color = code_colors[code_idx]
            full_frame[code_bar_y : code_bar_y + bar_height - 2, x_start:x_end] = color

        # Playhead for code bar
        playhead_x = int(i * width / n_frames)
        full_frame[
            code_bar_y : code_bar_y + bar_height - 2, playhead_x : playhead_x + 2
        ] = [255, 255, 255]

        # Separator
        full_frame[code_bar_y + bar_height - 2 : code_bar_y + bar_height, :] = [
            50,
            50,
            50,
        ]

        # === Bottom bar: Community timeline ===
        comm_bar_y = code_bar_y + bar_height
        for j in range(n_frames):
            x_start = int(j * width / n_frames)
            x_end = int((j + 1) * width / n_frames)
            code_idx = int(indices[j]) if j < len(indices) else 0
            comm_id = analysis.communities.get(code_idx, 0)
            color = comm_colors[comm_id]
            full_frame[comm_bar_y : comm_bar_y + bar_height - 2, x_start:x_end] = color

        # Playhead for community bar
        full_frame[
            comm_bar_y : comm_bar_y + bar_height - 2, playhead_x : playhead_x + 2
        ] = [255, 255, 255]

        frames.append(full_frame)

    renderer.close()

    # Write video
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return str(output_path)


def run_per_clip_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    num_clips: int = 10,
    n_communities: int | None = None,
    render_videos: bool = False,
    env: Any = None,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> dict[str, Any]:
    """Run per-clip analysis pipeline.

    Args:
        results: List of InferenceResult from rollouts.
        num_codes: Total number of codes in codebook.
        output_dir: Directory to save outputs.
        num_clips: Number of clips to analyze.
        n_communities: Number of communities per clip (None = auto-detect).
        render_videos: Whether to render videos for each clip.
        env: Environment for rendering (required if render_videos=True).
        camera: Camera name for video rendering.
        width: Video width.
        height: Video height.
        fps: Video frames per second.

    Returns:
        Dictionary with html_path, json_path, video_paths, and analyses.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips_to_analyze = list(results[:num_clips])
    logging.info(f"Running per-clip analysis on {len(clips_to_analyze)} clips...")

    analyses: list[PerClipAnalysis] = []
    for result in clips_to_analyze:
        analysis = analyze_single_clip(result, num_codes, n_communities)
        analyses.append(analysis)

        # Log community distribution
        comm_counts = {}
        for code, comm in analysis.communities.items():
            comm_counts[comm] = comm_counts.get(comm, 0) + 1
        comm_dist = ", ".join(f"C{c}:{n}" for c, n in sorted(comm_counts.items()))
        logging.info(
            f"  Clip {analysis.clip_idx}: {analysis.num_frames} frames, "
            f"{len(analysis.unique_codes)} codes, {analysis.n_communities} comms [{comm_dist}]"
        )

    # Generate interactive HTML viewer
    html_path = generate_interactive_html(
        analyses,
        num_codes,
        output_dir / "per_clip_analysis.html",
        title="Per-Clip VQ-VAE Analysis",
    )

    # Save JSON stats
    analyses_data = []
    for a in analyses:
        analyses_data.append(
            {
                "clip_idx": a.clip_idx,
                "num_frames": a.num_frames,
                "unique_codes": a.unique_codes,
                "n_communities": a.n_communities,
                "modularity": a.modularity,
                "code_frame_counts": a.code_frame_counts,
                "communities": a.communities,
            }
        )

    json_path = output_dir / "per_clip_stats.json"
    with open(json_path, "w") as f:
        json.dump(analyses_data, f, indent=2)

    # Render videos if requested
    video_paths = {}
    if render_videos and env is not None:
        logging.info("\nRendering clip videos with code/community bars...")
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        for i, (result, analysis) in enumerate(zip(clips_to_analyze, analyses)):
            if result.qpos is None:
                logging.warning(
                    f"  Clip {analysis.clip_idx}: No qpos data, skipping video"
                )
                continue

            video_path = video_dir / f"clip_{analysis.clip_idx:03d}.mp4"
            logging.info(
                f"  Rendering clip {analysis.clip_idx} ({i+1}/{len(analyses)})..."
            )

            try:
                path = render_clip_video(
                    env=env,
                    result=result,
                    analysis=analysis,
                    output_path=video_path,
                    num_codes=num_codes,
                    camera=camera,
                    width=width,
                    height=height,
                    fps=fps,
                )
                if path:
                    video_paths[f"clip_{analysis.clip_idx:03d}"] = path
            except Exception as e:
                logging.warning(f"  Failed to render clip {analysis.clip_idx}: {e}")

        logging.info(f"  Rendered {len(video_paths)} videos to {video_dir}")

    return {
        "analyses": analyses,
        "html_path": html_path,
        "json_path": str(json_path),
        "video_paths": video_paths,
    }
