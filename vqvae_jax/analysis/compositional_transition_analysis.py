"""Compositional Code Transition Analysis.

Analyzes whether VQ-VAE code transition sequences are deterministic and
compositional:

- **Deterministic**: Same starting pose + same code progression produces
  similar qpos trajectories (measured via Wasserstein-2 distance).
- **Compositional**: Longer code sequences decompose into independently
  observed shorter sub-sequences.

Key concepts:
- Code run: A contiguous block of the same code (e.g., [5,5,5] = one run
  of code 5).
- k-transition: A sequence of k+1 code runs representing k code changes
  (e.g., [5, 12, 7] is a 2-transition).
- Grace window: Transition boundaries can shift +/- N frames when comparing
  two sequences.
- W2 distance: Wasserstein-2 distance between qpos trajectories (per-joint
  Gaussian, RMS aggregated).
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CodeRun:
    """A contiguous block of the same code within a clip."""

    code: int
    start_frame: int
    end_frame: int  # exclusive


@dataclass
class TransitionSequence:
    """A sequence of k+1 code runs extracted from a clip."""

    clip_idx: int
    k: int  # number of transitions (len(runs) - 1)
    runs: list[CodeRun]
    code_sequence: tuple[int, ...]  # k+1 codes
    start_frame: int
    end_frame: int  # exclusive
    qpos_start: np.ndarray  # qpos at start_frame, joints only (excluding root 7 DOF)
    qpos_end: np.ndarray  # qpos at end_frame-1, joints only


@dataclass
class DeterminismResult:
    """A pair of matching transition sequences with their W2 distance."""

    seq_a: TransitionSequence
    seq_b: TransitionSequence
    start_qpos_distance: float
    w2_distance: float
    k: int
    code_sequence: tuple[int, ...]


@dataclass
class DecompositionNode:
    """A node in the compositional decomposition tree."""

    code_sequence: tuple[int, ...]
    k: int
    n_occurrences: int
    determinism_score: float | None  # mean W2 across pairs
    n_determinism_pairs: int
    children: list["DecompositionNode"] = field(default_factory=list)
    child_ranges: list[tuple[int, int]] = field(
        default_factory=list
    )  # (start_pos, end_pos) in parent
    is_leaf: bool = True


# =============================================================================
# EXTRACTION
# =============================================================================


def extract_code_runs(code_indices: np.ndarray) -> list[CodeRun]:
    """Extract contiguous code runs from a sequence of code indices.

    Args:
        code_indices: Array of shape [T] with discrete code per frame.

    Returns:
        List of CodeRun objects representing contiguous blocks.
    """
    if len(code_indices) == 0:
        return []

    runs = []
    current_code = int(code_indices[0])
    start = 0

    for i in range(1, len(code_indices)):
        if int(code_indices[i]) != current_code:
            runs.append(CodeRun(code=current_code, start_frame=start, end_frame=i))
            current_code = int(code_indices[i])
            start = i

    # Final run
    runs.append(
        CodeRun(code=current_code, start_frame=start, end_frame=len(code_indices))
    )

    return runs


def extract_transition_sequences(
    result: InferenceResult,
    min_k: int,
    max_k: int,
) -> list[TransitionSequence]:
    """Extract all k-transition sequences from a single clip.

    Slides a window of size k+1 over the code runs to extract all
    sub-sequences for k from min_k to max_k.

    Args:
        result: InferenceResult for one clip.
        min_k: Minimum number of transitions.
        max_k: Maximum number of transitions.

    Returns:
        List of TransitionSequence objects.
    """
    runs = extract_code_runs(result.code_indices)
    if len(runs) < min_k + 1:
        return []

    sequences = []
    for k in range(min_k, max_k + 1):
        window_size = k + 1
        if len(runs) < window_size:
            break

        for i in range(len(runs) - window_size + 1):
            sub_runs = runs[i : i + window_size]
            code_seq = tuple(r.code for r in sub_runs)
            start_frame = sub_runs[0].start_frame
            end_frame = sub_runs[-1].end_frame

            # Extract joint qpos (exclude root 7 DOF)
            qpos_start = result.qpos[start_frame, 7:].copy()
            qpos_end = result.qpos[end_frame - 1, 7:].copy()

            sequences.append(
                TransitionSequence(
                    clip_idx=result.clip_idx,
                    k=k,
                    runs=sub_runs,
                    code_sequence=code_seq,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    qpos_start=qpos_start,
                    qpos_end=qpos_end,
                )
            )

    return sequences


def build_transition_index(
    results: Sequence[InferenceResult],
    min_k: int,
    max_k: int,
) -> dict[int, dict[tuple[int, ...], list[TransitionSequence]]]:
    """Build an index of all transition sequences grouped by k and code sequence.

    Args:
        results: List of InferenceResult objects.
        min_k: Minimum number of transitions.
        max_k: Maximum number of transitions.

    Returns:
        Nested dict: {k: {code_sequence_tuple: [TransitionSequence, ...]}}.
    """
    index: dict[int, dict[tuple[int, ...], list[TransitionSequence]]] = {}
    for k in range(min_k, max_k + 1):
        index[k] = {}

    for result in results:
        sequences = extract_transition_sequences(result, min_k, max_k)
        for seq in sequences:
            group = index[seq.k].setdefault(seq.code_sequence, [])
            group.append(seq)

    return index


# =============================================================================
# MATCHING & DISTANCE
# =============================================================================


def check_boundary_alignment(
    seq_a: TransitionSequence,
    seq_b: TransitionSequence,
    grace_window: int,
) -> bool:
    """Check whether transition boundaries are aligned within grace window.

    Compares the relative boundary positions (normalized to sequence length)
    between two sequences.

    Args:
        seq_a: First transition sequence.
        seq_b: Second transition sequence.
        grace_window: Maximum allowed frame difference per boundary.

    Returns:
        True if all boundaries align within the grace window.
    """
    if len(seq_a.runs) != len(seq_b.runs):
        return False

    # Compare relative boundary positions
    len_a = seq_a.end_frame - seq_a.start_frame
    len_b = seq_b.end_frame - seq_b.start_frame

    if len_a == 0 or len_b == 0:
        return False

    for i in range(1, len(seq_a.runs)):
        # Relative position of boundary i in each sequence
        boundary_a = seq_a.runs[i].start_frame - seq_a.start_frame
        boundary_b = seq_b.runs[i].start_frame - seq_b.start_frame

        # Scale to common reference (use average length)
        avg_len = (len_a + len_b) / 2.0
        scaled_a = boundary_a * avg_len / len_a
        scaled_b = boundary_b * avg_len / len_b

        if abs(scaled_a - scaled_b) > grace_window:
            return False

    return True


def compute_w2_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """Compute Wasserstein-2 distance between two qpos trajectories.

    Per-joint: fit Gaussian (mean, std), compute W2^2 = (mu_a - mu_b)^2 +
    (sigma_a - sigma_b)^2. Return sqrt(mean(W2_j^2)) across joints.

    Args:
        traj_a: Qpos trajectory, shape [T_a, n_joints].
        traj_b: Qpos trajectory, shape [T_b, n_joints].

    Returns:
        RMS W2 distance across joints.
    """
    # Resample both to 50 time points
    n_points = 50
    n_joints = traj_a.shape[1]

    def resample(traj: np.ndarray) -> np.ndarray:
        t_orig = np.linspace(0, 1, len(traj))
        t_new = np.linspace(0, 1, n_points)
        resampled = np.zeros((n_points, traj.shape[1]))
        for j in range(traj.shape[1]):
            resampled[:, j] = np.interp(t_new, t_orig, traj[:, j])
        return resampled

    traj_a_r = resample(traj_a)
    traj_b_r = resample(traj_b)

    # Per-joint W2 distance
    w2_sq = np.zeros(n_joints)
    for j in range(n_joints):
        mu_a, sigma_a = np.mean(traj_a_r[:, j]), np.std(traj_a_r[:, j])
        mu_b, sigma_b = np.mean(traj_b_r[:, j]), np.std(traj_b_r[:, j])
        w2_sq[j] = (mu_a - mu_b) ** 2 + (sigma_a - sigma_b) ** 2

    return float(np.sqrt(np.mean(w2_sq)))


def extract_qpos_trajectory(
    result: InferenceResult,
    seq: TransitionSequence,
) -> np.ndarray:
    """Extract joint qpos trajectory for a transition sequence.

    Args:
        result: InferenceResult containing the clip data.
        seq: TransitionSequence specifying the frame range.

    Returns:
        Array of shape [T, n_joints] with joint angles (excluding root 7 DOF).
    """
    return result.qpos[seq.start_frame : seq.end_frame, 7:]


# =============================================================================
# DETERMINISM CHECK
# =============================================================================


def find_determinism_pairs(
    index: dict[int, dict[tuple[int, ...], list[TransitionSequence]]],
    results_by_clip: dict[int, InferenceResult],
    qpos_threshold: float,
    grace_window: int,
    min_k: int,
    max_k: int,
) -> dict[int, list[DeterminismResult]]:
    """Find all cross-clip determinism pairs and compute W2 distances.

    For each code sequence group with >=2 sequences from different clips,
    checks starting qpos similarity and boundary alignment, then computes
    W2 distance.

    Args:
        index: Transition index from build_transition_index().
        results_by_clip: Dict mapping clip_idx to InferenceResult.
        qpos_threshold: Max mean abs diff for starting qpos match.
        grace_window: Boundary alignment tolerance in frames.
        min_k: Minimum transitions.
        max_k: Maximum transitions.

    Returns:
        Dict mapping k to list of DeterminismResult.
    """
    pairs: dict[int, list[DeterminismResult]] = {}
    for k in range(min_k, max_k + 1):
        pairs[k] = []

    for k in range(min_k, max_k + 1):
        for code_seq, seqs in index[k].items():
            if len(seqs) < 2:
                continue

            # Only compare cross-clip pairs
            for i in range(len(seqs)):
                for j in range(i + 1, len(seqs)):
                    seq_a, seq_b = seqs[i], seqs[j]
                    if seq_a.clip_idx == seq_b.clip_idx:
                        continue

                    # Check starting qpos similarity
                    qpos_dist = float(
                        np.mean(np.abs(seq_a.qpos_start - seq_b.qpos_start))
                    )
                    if qpos_dist >= qpos_threshold:
                        continue

                    # Check boundary alignment
                    if not check_boundary_alignment(seq_a, seq_b, grace_window):
                        continue

                    # Compute W2 distance
                    result_a = results_by_clip[seq_a.clip_idx]
                    result_b = results_by_clip[seq_b.clip_idx]
                    traj_a = extract_qpos_trajectory(result_a, seq_a)
                    traj_b = extract_qpos_trajectory(result_b, seq_b)

                    if len(traj_a) < 2 or len(traj_b) < 2:
                        continue

                    w2 = compute_w2_distance(traj_a, traj_b)

                    pairs[k].append(
                        DeterminismResult(
                            seq_a=seq_a,
                            seq_b=seq_b,
                            start_qpos_distance=qpos_dist,
                            w2_distance=w2,
                            k=k,
                            code_sequence=code_seq,
                        )
                    )

    return pairs


def summarize_determinism(
    pairs: dict[int, list[DeterminismResult]],
) -> dict[str, Any]:
    """Summarize determinism results by k level.

    Args:
        pairs: Dict mapping k to list of DeterminismResult.

    Returns:
        Summary dict with per-k statistics.
    """
    summary: dict[str, Any] = {"by_k": {}}

    for k, pair_list in sorted(pairs.items()):
        if not pair_list:
            summary["by_k"][k] = {
                "n_pairs": 0,
                "mean_w2": None,
                "median_w2": None,
                "std_w2": None,
                "n_unique_sequences": 0,
            }
            continue

        w2_values = [p.w2_distance for p in pair_list]
        unique_seqs = set(p.code_sequence for p in pair_list)

        summary["by_k"][k] = {
            "n_pairs": len(pair_list),
            "mean_w2": float(np.mean(w2_values)),
            "median_w2": float(np.median(w2_values)),
            "std_w2": float(np.std(w2_values)),
            "n_unique_sequences": len(unique_seqs),
        }

    return summary


def plot_determinism_by_k(
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Plot determinism (W2 distance) as a function of k.

    Args:
        summary: Output from summarize_determinism().
        output_dir: Directory to save figures.

    Returns:
        Dict mapping figure name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    by_k = summary["by_k"]
    ks = sorted(k for k, v in by_k.items() if v["n_pairs"] > 0)

    if not ks:
        logging.warning("No determinism pairs found to plot")
        return paths

    means = [by_k[k]["mean_w2"] for k in ks]
    medians = [by_k[k]["median_w2"] for k in ks]
    stds = [by_k[k]["std_w2"] for k in ks]
    n_pairs = [by_k[k]["n_pairs"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: W2 distance vs k
    ax = axes[0]
    ax.errorbar(ks, means, yerr=stds, fmt="o-", capsize=5, label="Mean +/- Std")
    ax.plot(ks, medians, "s--", color="orange", label="Median")
    ax.set_xlabel("k (number of transitions)")
    ax.set_ylabel("W2 Distance")
    ax.set_title("Trajectory Determinism by Transition Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Number of pairs vs k
    ax = axes[1]
    ax.bar(ks, n_pairs, color="steelblue", alpha=0.7)
    ax.set_xlabel("k (number of transitions)")
    ax.set_ylabel("Number of Valid Pairs")
    ax.set_title("Cross-clip Determinism Pair Counts")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = output_dir / "determinism_by_k.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["determinism_by_k"] = str(fig_path)

    return paths


# =============================================================================
# COMPOSITIONAL DECOMPOSITION
# =============================================================================


def build_decomposition_trees(
    index: dict[int, dict[tuple[int, ...], list[TransitionSequence]]],
    pairs: dict[int, list[DeterminismResult]],
    min_k: int,
    max_k: int,
) -> dict[int, list[DecompositionNode]]:
    """Build decomposition trees for all observed code sequences.

    Uses bottom-up construction with memoization: processes k=min_k first,
    then tries to decompose longer sequences using cached sub-sequences.

    Args:
        index: Transition index from build_transition_index().
        pairs: Determinism pairs from find_determinism_pairs().
        min_k: Minimum transitions.
        max_k: Maximum transitions.

    Returns:
        Dict mapping k to list of DecompositionNode (sorted by occurrence
        count descending).
    """
    # Precompute determinism scores per code sequence
    det_scores: dict[tuple[int, ...], tuple[float, int]] = {}
    for k_pairs in pairs.values():
        for pair in k_pairs:
            key = pair.code_sequence
            if key not in det_scores:
                det_scores[key] = (0.0, 0)
            total, count = det_scores[key]
            det_scores[key] = (total + pair.w2_distance, count + 1)

    # Cache of all built nodes
    cache: dict[tuple[int, ...], DecompositionNode] = {}

    trees: dict[int, list[DecompositionNode]] = {}

    for k in range(min_k, max_k + 1):
        trees[k] = []

        for code_seq, seqs in index[k].items():
            n_occ = len(seqs)

            # Compute determinism score
            det_score = None
            n_det_pairs = 0
            if code_seq in det_scores:
                total, count = det_scores[code_seq]
                det_score = total / count if count > 0 else None
                n_det_pairs = count

            node = DecompositionNode(
                code_sequence=code_seq,
                k=k,
                n_occurrences=n_occ,
                determinism_score=det_score,
                n_determinism_pairs=n_det_pairs,
            )

            # Try to decompose at each valid split position
            # Split at position p means:
            #   left = code_seq[0:p+1] (p transitions)
            #   right = code_seq[p:]   (k-p transitions)
            best_children: list[DecompositionNode] = []
            best_ranges: list[tuple[int, int]] = []

            for p in range(min_k, k - min_k + 1):
                left_seq = code_seq[: p + 1]
                right_seq = code_seq[p:]
                left_k = p
                right_k = k - p

                if left_seq in cache and right_seq in cache:
                    left_node = cache[left_seq]
                    right_node = cache[right_seq]

                    children = [left_node, right_node]
                    ranges = [(0, p), (p, k)]

                    # Prefer split that maximizes total sub-occurrences
                    total_sub_occ = left_node.n_occurrences + right_node.n_occurrences
                    best_sub_occ = sum(c.n_occurrences for c in best_children)

                    if total_sub_occ > best_sub_occ:
                        best_children = children
                        best_ranges = ranges

            if best_children:
                node.children = best_children
                node.child_ranges = best_ranges
                node.is_leaf = False

            cache[code_seq] = node
            trees[k].append(node)

        # Sort by occurrence count descending
        trees[k].sort(key=lambda n: n.n_occurrences, reverse=True)

    return trees


def select_top_sequences(
    index: dict[int, dict[tuple[int, ...], list[TransitionSequence]]],
    top_n: int,
) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Select top-N most popular code sequences per k level.

    Args:
        index: Transition index.
        top_n: Number of sequences to select per k.

    Returns:
        Dict mapping k to list of (code_sequence, count) tuples sorted by
        count descending.
    """
    result: dict[int, list[tuple[tuple[int, ...], int]]] = {}
    for k, groups in index.items():
        ranked = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        result[k] = [(seq, len(seqs)) for seq, seqs in ranked[:top_n]]
    return result


# =============================================================================
# HTML VISUALIZATION
# =============================================================================


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


def _get_code_color(code: int, num_codes: int) -> str:
    """Get a hex color for a given code index."""
    if num_codes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis
    rgba = cmap(code / max(num_codes - 1, 1))
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _node_to_html(
    node: DecompositionNode,
    num_codes: int,
    depth: int = 0,
    videos_b64: dict[tuple[int, ...], str] | None = None,
) -> str:
    """Recursively render a DecompositionNode as HTML.

    Args:
        node: The decomposition node.
        num_codes: Total number of codes (for coloring).
        depth: Current recursion depth.
        videos_b64: Optional dict mapping code_sequence to base64 video data URL.

    Returns:
        HTML string for this node and its children.
    """
    # Code badges
    badges = " ".join(
        f'<span class="code-badge" style="background:{_get_code_color(c, num_codes)}">'
        f"{c}</span>"
        for c in node.code_sequence
    )

    # Stats
    det_str = (
        f"{node.determinism_score:.4f}" if node.determinism_score is not None else "N/A"
    )
    stats = (
        f'<span class="stat">k={node.k}</span>'
        f'<span class="stat">n={node.n_occurrences}</span>'
        f'<span class="stat">W2={det_str}</span>'
        f'<span class="stat">pairs={node.n_determinism_pairs}</span>'
    )

    # Video element if available
    video_html = ""
    if videos_b64 and node.code_sequence in videos_b64:
        src = videos_b64[node.code_sequence]
        if src:
            video_html = (
                f'<video class="node-video" controls loop muted playsinline '
                f'width="320" height="240"><source src="{src}" '
                f'type="video/mp4"></video>'
            )

    # Children
    children_html = ""
    if node.children:
        child_items = []
        for i, (child, (start_pos, end_pos)) in enumerate(
            zip(node.children, node.child_ranges)
        ):
            range_label = f"positions {start_pos}-{end_pos}"
            child_html = _node_to_html(child, num_codes, depth + 1, videos_b64)
            child_items.append(
                f'<div class="child-wrapper">'
                f'<div class="range-label">{range_label}</div>'
                f"{child_html}"
                f"</div>"
            )
        children_html = (
            f'<div class="children" id="children-{id(node)}">'
            + "".join(child_items)
            + "</div>"
        )

    # Collapse/expand toggle
    toggle = ""
    if node.children:
        toggle = (
            f'<button class="toggle-btn" '
            f"onclick=\"toggleNode('children-{id(node)}', this)\">"
            f"[+]</button>"
        )

    leaf_class = "leaf" if node.is_leaf else "branch"

    return (
        f'<div class="tree-node {leaf_class}" style="margin-left:{depth * 24}px">'
        f'<div class="node-header">'
        f"{toggle}"
        f'<div class="code-badges">{badges}</div>'
        f'<div class="node-stats">{stats}</div>'
        f"</div>"
        f"{video_html}"
        f"{children_html}"
        f"</div>"
    )


def generate_code_tree_html(
    trees: dict[int, list[DecompositionNode]],
    num_codes: int,
    summary: dict[str, Any],
    top_n: int,
    output_path: str | Path,
    videos_b64: dict[tuple[int, ...], str] | None = None,
) -> str:
    """Generate the full HTML page for the compositional tree viewer.

    Args:
        trees: Decomposition trees from build_decomposition_trees().
        num_codes: Total number of codes.
        summary: Determinism summary from summarize_determinism().
        top_n: Number of sequences to show per k level.
        output_path: Path to save the HTML file.
        videos_b64: Optional dict of base64 video data URLs.

    Returns:
        Path to the saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build summary stats table
    summary_rows = ""
    by_k = summary.get("by_k", {})
    for k in sorted(by_k.keys()):
        s = by_k[k]
        n_pairs = s["n_pairs"]
        mean_w2 = f"{s['mean_w2']:.4f}" if s["mean_w2"] is not None else "N/A"
        median_w2 = f"{s['median_w2']:.4f}" if s["median_w2"] is not None else "N/A"
        std_w2 = f"{s['std_w2']:.4f}" if s["std_w2"] is not None else "N/A"
        n_seqs = s["n_unique_sequences"]
        summary_rows += (
            f"<tr><td>{k}</td><td>{n_pairs}</td><td>{mean_w2}</td>"
            f"<td>{median_w2}</td><td>{std_w2}</td><td>{n_seqs}</td></tr>"
        )

    # Build tree content per k level
    k_tabs = ""
    k_contents = ""
    sorted_ks = sorted(trees.keys())

    for k in sorted_ks:
        nodes = trees[k][:top_n]
        n_total = len(trees[k])
        n_composable = sum(1 for n in trees[k] if not n.is_leaf)

        k_tabs += (
            f'<button class="k-tab" onclick="showK({k})" id="tab-{k}">'
            f"k={k} ({n_total})</button>"
        )

        nodes_html = ""
        for node in nodes:
            nodes_html += _node_to_html(node, num_codes, 0, videos_b64)

        k_contents += (
            f'<div class="k-content" id="content-{k}" style="display:none">'
            f'<div class="k-summary">'
            f"<strong>k={k}</strong>: {n_total} unique sequences, "
            f"{n_composable} decomposable "
            f"({n_composable / max(n_total, 1) * 100:.1f}%)"
            f"</div>"
            f"{nodes_html}"
            f"</div>"
        )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Compositional Code Transition Analysis</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 20px;
}}
h1 {{
    text-align: center;
    font-size: 1.8em;
    margin-bottom: 20px;
    color: #64b5f6;
}}
h2 {{
    font-size: 1.3em;
    margin: 20px 0 10px;
    color: #90caf9;
}}
.summary-table {{
    width: 100%;
    max-width: 800px;
    margin: 0 auto 30px;
    border-collapse: collapse;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    overflow: hidden;
}}
.summary-table th, .summary-table td {{
    padding: 8px 16px;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}
.summary-table th {{
    background: rgba(100,181,246,0.2);
    color: #90caf9;
    font-weight: 600;
}}
.k-tabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
    justify-content: center;
}}
.k-tab {{
    padding: 8px 20px;
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    color: #90caf9;
    cursor: pointer;
    font-size: 0.9em;
    transition: all 0.2s;
}}
.k-tab:hover {{ background: rgba(100,181,246,0.2); }}
.k-tab.active {{
    background: rgba(100,181,246,0.3);
    border-color: #64b5f6;
    color: #fff;
}}
.k-content {{ margin-bottom: 30px; }}
.k-summary {{
    padding: 10px 16px;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    margin-bottom: 15px;
    font-size: 0.95em;
}}
.tree-node {{
    border-left: 2px solid rgba(100,181,246,0.3);
    padding: 8px 12px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 0 8px 8px 0;
    transition: background 0.2s;
}}
.tree-node:hover {{ background: rgba(255,255,255,0.07); }}
.tree-node.leaf {{ border-left-color: rgba(129,199,132,0.4); }}
.node-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}}
.toggle-btn {{
    background: none;
    border: 1px solid rgba(255,255,255,0.3);
    color: #90caf9;
    cursor: pointer;
    font-family: monospace;
    font-size: 0.85em;
    padding: 2px 6px;
    border-radius: 4px;
    min-width: 30px;
}}
.toggle-btn:hover {{ background: rgba(100,181,246,0.2); }}
.code-badges {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.code-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
    color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    min-width: 30px;
    text-align: center;
}}
.node-stats {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}
.stat {{
    font-size: 0.8em;
    padding: 2px 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    color: #aaa;
}}
.children {{
    display: none;
    margin-top: 8px;
    padding-left: 12px;
}}
.child-wrapper {{ margin-bottom: 4px; }}
.range-label {{
    font-size: 0.75em;
    color: #888;
    margin-bottom: 2px;
    font-style: italic;
}}
.node-video {{
    margin-top: 8px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}}
</style>
</head>
<body>
<h1>Compositional Code Transition Analysis</h1>

<h2>Determinism Summary</h2>
<table class="summary-table">
<thead>
<tr><th>k</th><th>Pairs</th><th>Mean W2</th><th>Median W2</th><th>Std W2</th><th>Unique Seqs</th></tr>
</thead>
<tbody>{summary_rows}</tbody>
</table>

<h2>Decomposition Trees</h2>
<div class="k-tabs">{k_tabs}</div>
{k_contents}

<script>
function toggleNode(childId, btn) {{
    var el = document.getElementById(childId);
    if (el.style.display === 'none' || el.style.display === '') {{
        el.style.display = 'block';
        btn.textContent = '[-]';
    }} else {{
        el.style.display = 'none';
        btn.textContent = '[+]';
    }}
}}

function showK(k) {{
    // Hide all k-contents
    document.querySelectorAll('.k-content').forEach(function(el) {{
        el.style.display = 'none';
    }});
    // Deactivate all tabs
    document.querySelectorAll('.k-tab').forEach(function(el) {{
        el.classList.remove('active');
    }});
    // Show selected
    var content = document.getElementById('content-' + k);
    if (content) content.style.display = 'block';
    var tab = document.getElementById('tab-' + k);
    if (tab) tab.classList.add('active');
}}

// Show first k by default
var firstK = {sorted_ks[0] if sorted_ks else 2};
showK(firstK);
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    logging.info(f"Saved compositional tree HTML to {output_path}")
    return str(output_path)


# =============================================================================
# VIDEO RENDERING
# =============================================================================


def render_subsequence_video(
    env: Any,
    result: InferenceResult,
    seq: TransitionSequence,
    output_path: str | Path,
    num_codes: int,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> str | None:
    """Render a video of a specific transition sequence.

    Args:
        env: Environment with mj_model for rendering.
        result: InferenceResult containing clip data.
        seq: TransitionSequence specifying frames to render.
        output_path: Path to save the video.
        num_codes: Number of codes for colormap.
        camera: Camera name.
        width: Video width.
        height: Video height.
        fps: Frames per second.

    Returns:
        Path to the saved video, or None on failure.
    """
    try:
        import imageio
        import mujoco
    except ImportError:
        logging.warning("Could not import mujoco/imageio for video rendering")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if result.qpos is None or len(result.qpos) == 0:
        logging.warning(f"No qpos for clip {result.clip_idx}, skipping video")
        return None

    try:
        from .rendering import get_nature_colormap

        code_colors = get_nature_colormap(num_codes)
    except ImportError:
        code_colors = None

    bar_height = 30

    try:
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
                pass

        sub_qpos = result.qpos[seq.start_frame : seq.end_frame]
        sub_indices = result.code_indices[seq.start_frame : seq.end_frame]
        n_frames = len(sub_qpos)

        frames = []
        for i in range(n_frames):
            mj_data.qpos[:] = sub_qpos[i]
            mujoco.mj_forward(mj_model, mj_data)

            if cam_id >= 0:
                renderer.update_scene(mj_data, camera=cam_id)
            else:
                renderer.update_scene(mj_data)
            render_frame = renderer.render()

            # Build full frame with code timeline bar
            full_frame = np.ones((height, width, 3), dtype=np.uint8) * 40
            full_frame[:render_height, :] = render_frame

            # Code timeline bar
            if code_colors is not None:
                for j in range(n_frames):
                    x_start = int(j * width / n_frames)
                    x_end = int((j + 1) * width / n_frames)
                    code_idx = int(sub_indices[j])
                    full_frame[
                        render_height : render_height + bar_height - 2,
                        x_start:x_end,
                    ] = code_colors[code_idx]

                # Playhead
                px = int(i * width / n_frames)
                full_frame[
                    render_height : render_height + bar_height - 2, px : px + 2
                ] = [255, 255, 255]

            frames.append(full_frame)

        renderer.close()

        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

        return str(output_path)

    except Exception as e:
        logging.warning(f"Failed to render subsequence video: {e}")
        return None


# =============================================================================
# PIPELINE ENTRY POINTS
# =============================================================================


def run_qpos_code_determinism_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the qpos+code determinism analysis.

    Tests whether the same starting pose + same code progression produces
    similar qpos trajectories.

    Args:
        results: List of InferenceResult objects.
        num_codes: Number of codes in the codebook.
        output_dir: Directory for output files.
        cfg: Configuration dict with keys: num_clips, min_k, max_k,
            grace_window, qpos_threshold.

    Returns:
        Dict with output paths and summary data.
    """
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_clips = cfg.get("num_clips", 200)
    min_k = cfg.get("min_k", 2)
    max_k = cfg.get("max_k", 8)
    grace_window = cfg.get("grace_window", 10)
    qpos_threshold = cfg.get("qpos_threshold", 0.05)

    # Select clips
    selected = list(results[:num_clips])
    logging.info(f"Determinism analysis: {len(selected)} clips, k={min_k}-{max_k}")

    # Build index
    logging.info("  Building transition index...")
    index = build_transition_index(selected, min_k, max_k)

    total_seqs = sum(len(seqs) for groups in index.values() for seqs in groups.values())
    total_groups = sum(len(groups) for groups in index.values())
    logging.info(f"  {total_seqs} sequences in {total_groups} groups")

    # Build clip lookup
    results_by_clip = {r.clip_idx: r for r in selected}

    # Find determinism pairs
    logging.info("  Finding determinism pairs...")
    pairs = find_determinism_pairs(
        index, results_by_clip, qpos_threshold, grace_window, min_k, max_k
    )
    total_pairs = sum(len(p) for p in pairs.values())
    logging.info(f"  Found {total_pairs} valid pairs")

    # Summarize
    summary = summarize_determinism(pairs)

    # Plot
    fig_paths = plot_determinism_by_k(summary, output_dir)

    # Save JSON summary
    json_path = output_dir / "determinism_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "summary": summary,
        "figure_paths": fig_paths,
        "json_path": str(json_path),
        "total_pairs": total_pairs,
    }


def run_compositional_transition_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: Path,
    cfg: dict[str, Any] | None = None,
    env: Any | None = None,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> dict[str, Any]:
    """Run the full compositional transition analysis.

    Builds decomposition trees showing how k-transitions break down into
    sub-sequences, with determinism scores at each level.

    Args:
        results: List of InferenceResult objects.
        num_codes: Number of codes in the codebook.
        output_dir: Directory for output files.
        cfg: Configuration dict with keys: num_clips, min_k, max_k,
            grace_window, qpos_threshold, top_n_sequences, render_videos.
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

    num_clips = cfg.get("num_clips", 200)
    min_k = cfg.get("min_k", 2)
    max_k = cfg.get("max_k", 8)
    grace_window = cfg.get("grace_window", 10)
    qpos_threshold = cfg.get("qpos_threshold", 0.05)
    top_n = cfg.get("top_n_sequences", 20)
    render_videos = cfg.get("render_videos", False)

    # Select clips
    selected = list(results[:num_clips])
    logging.info(f"Compositional analysis: {len(selected)} clips, k={min_k}-{max_k}")

    # Build index
    logging.info("  Building transition index...")
    index = build_transition_index(selected, min_k, max_k)

    total_seqs = sum(len(seqs) for groups in index.values() for seqs in groups.values())
    total_groups = sum(len(groups) for groups in index.values())
    logging.info(f"  {total_seqs} sequences in {total_groups} groups")

    # Build clip lookup
    results_by_clip = {r.clip_idx: r for r in selected}

    # Find determinism pairs
    logging.info("  Finding determinism pairs...")
    pairs = find_determinism_pairs(
        index, results_by_clip, qpos_threshold, grace_window, min_k, max_k
    )
    total_pairs = sum(len(p) for p in pairs.values())
    logging.info(f"  Found {total_pairs} valid pairs")

    # Summarize determinism
    summary = summarize_determinism(pairs)

    # Build decomposition trees
    logging.info("  Building decomposition trees...")
    trees = build_decomposition_trees(index, pairs, min_k, max_k)

    for k in sorted(trees.keys()):
        n_total = len(trees[k])
        n_composable = sum(1 for n in trees[k] if not n.is_leaf)
        logging.info(f"    k={k}: {n_total} sequences, {n_composable} decomposable")

    # Render videos if enabled
    videos_b64: dict[tuple[int, ...], str] | None = None
    if render_videos and env is not None:
        logging.info("  Rendering example videos...")
        videos_b64 = {}
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        for k in sorted(trees.keys()):
            for node in trees[k][:top_n]:
                code_seq = node.code_sequence
                # Find the first occurrence to render
                if code_seq in index[k] and index[k][code_seq]:
                    seq = index[k][code_seq][0]
                    clip_result = results_by_clip.get(seq.clip_idx)
                    if clip_result is None:
                        continue

                    seq_name = "_".join(str(c) for c in code_seq)
                    video_path = video_dir / f"k{k}_{seq_name}.mp4"
                    result_path = render_subsequence_video(
                        env=env,
                        result=clip_result,
                        seq=seq,
                        output_path=video_path,
                        num_codes=num_codes,
                        camera=camera,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    if result_path:
                        videos_b64[code_seq] = video_to_base64(result_path)

        logging.info(f"  Rendered {len(videos_b64)} videos")

    # Generate HTML viewer
    html_path = output_dir / "compositional_tree.html"
    generate_code_tree_html(
        trees=trees,
        num_codes=num_codes,
        summary=summary,
        top_n=top_n,
        output_path=html_path,
        videos_b64=videos_b64,
    )

    # Save JSON summary
    json_summary = {
        "determinism": summary,
        "composition": {},
    }
    for k in sorted(trees.keys()):
        n_total = len(trees[k])
        n_composable = sum(1 for n in trees[k] if not n.is_leaf)
        json_summary["composition"][str(k)] = {
            "n_total": n_total,
            "n_composable": n_composable,
            "composable_pct": n_composable / max(n_total, 1) * 100,
            "top_sequences": [
                {
                    "code_sequence": list(n.code_sequence),
                    "n_occurrences": n.n_occurrences,
                    "determinism_score": n.determinism_score,
                    "n_determinism_pairs": n.n_determinism_pairs,
                    "is_composable": not n.is_leaf,
                }
                for n in trees[k][:top_n]
            ],
        }

    json_path = output_dir / "compositional_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
        "summary": json_summary,
        "total_pairs": total_pairs,
    }
