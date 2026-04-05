"""Visualize KPMS syllable filmstrips — code sequences mapped to behavior poses.

For each sample clip, produces a horizontal figure:
  Top:    colored code bar showing the syllable sequence over time
  Bottom: 3-pose filmstrips for each syllable, connected by arrows

Uses the same tinted-pose rendering style as the TopoVNL filmstrip figures.

Usage:
    cd moseq_jax
    python visualize_kpms_filmstrips.py
    python visualize_kpms_filmstrips.py --clips 1 6 17 --output outputs/kpms_filmstrips
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import math
from pathlib import Path

import h5py
import mujoco
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CODES = str(REPO_ROOT / "moseq_jax/outputs/kpms_sweep/best_codes.npz")
DEFAULT_H5 = str(REPO_ROOT / "data/rodent/rodent_reference_clips.h5")

_XML_PATH = None  # resolved lazily
# The stac-mjx rodent XML has a freejoint (nq=74) matching the h5 qpos data.
# The vnl_playground XML omits the freejoint (nq=67) — unusable for raw qpos.
_STAC_XML = "/home/jovyan/vast/kaiwen/TopoVNL/stac-mjx/models/rodent.xml"


def _get_xml_path() -> str:
    global _XML_PATH
    if _XML_PATH is None:
        if os.path.exists(_STAC_XML):
            _XML_PATH = _STAC_XML
        else:
            import vnl_playground

            _XML_PATH = os.path.join(
                os.path.dirname(vnl_playground.__file__),
                "tasks/rodent/xmls/rodent.xml",
            )
    return _XML_PATH


# ── Color palette ────────────────────────────────────────────────────────────
_CODE_PALETTE = [
    "#5B83AC", "#E89850", "#55A868", "#CC6677", "#9070A0",
    "#C0A0D0", "#4C72B0", "#DD8452", "#7BC47F", "#8172B2",
    "#CCB974", "#64B5CD", "#D65F5F", "#4878A8", "#E8A850",
    "#6ACC64", "#C44E52", "#8DA0CB", "#FC8D62", "#66C2A5",
]


def _code_color_map(codes_flat: np.ndarray) -> dict[int, str]:
    unique = sorted(set(int(c) for c in codes_flat))
    return {c: _CODE_PALETTE[i % len(_CODE_PALETTE)] for i, c in enumerate(unique)}


# ── Data helpers ─────────────────────────────────────────────────────────────


def find_syllable_segments(
    codes: np.ndarray, min_frames: int = 5
) -> list[tuple[int, int, int]]:
    """Find contiguous runs of the same code.

    Returns:
        List of ``(code, start_frame, end_frame)`` with length >= *min_frames*.
    """
    segments: list[tuple[int, int, int]] = []
    cur = int(codes[0])
    start = 0
    for i in range(1, len(codes)):
        if int(codes[i]) != cur:
            if i - start >= min_frames:
                segments.append((cur, start, i))
            cur = int(codes[i])
            start = i
    if len(codes) - start >= min_frames:
        segments.append((cur, start, len(codes)))
    return segments


# ── MuJoCo rendering ────────────────────────────────────────────────────────


def _render_cropped_poses(
    qpos_clip: np.ndarray,
    frame_indices: list[int],
    xml_path: str,
    fixed_crop: tuple[int, int, int, int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Render frames and auto-crop to rodent body bounds.

    Returns:
        List of ``(cropped_rgb, body_mask)`` tuples.
    """
    fovy_rad = math.radians(45.0)
    view_h = 0.35
    dist = (view_h / 2.0) / math.tan(fovy_rad / 2.0) * 1.3
    render_h, render_w = 800, 800

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    rgb_renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    seg_renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    seg_renderer.enable_segmentation_rendering()

    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = dist
    camera.elevation = -35.0
    camera.azimuth = 90.0

    _blank_rgb = np.full((100, 60, 3), 255, dtype=np.uint8)
    _blank_mask = np.zeros((100, 60), dtype=bool)

    # Fixed crop size (same for all frames/clips so rodents are uniform)
    if fixed_crop is not None:
        crop_half_h = (fixed_crop[1] - fixed_crop[0]) // 2
        crop_half_w = (fixed_crop[3] - fixed_crop[2]) // 2
    else:
        crop_half_h, crop_half_w = 200, 200  # fallback

    crop_h = 2 * crop_half_h
    crop_w = 2 * crop_half_w

    results: list[tuple[np.ndarray, np.ndarray]] = []
    for fi in frame_indices:
        if fi >= qpos_clip.shape[0]:
            results.append((
                np.full((crop_h, crop_w, 3), 255, dtype=np.uint8),
                np.zeros((crop_h, crop_w), dtype=bool),
            ))
            continue

        camera.lookat[:] = [
            qpos_clip[fi, 0],
            qpos_clip[fi, 1],
            qpos_clip[fi, 2] + 0.02,
        ]
        mj_data.qpos[:] = qpos_clip[fi]
        mujoco.mj_forward(mj_model, mj_data)

        rgb_renderer.update_scene(mj_data, camera=camera)
        rgb = rgb_renderer.render().copy()

        seg_renderer.update_scene(mj_data, camera=camera)
        seg = seg_renderer.render()
        geom_ids = seg[:, :, 0].astype(np.int32)
        body_mask = (geom_ids >= 0) & (geom_ids != floor_id)

        rows = np.any(body_mask, axis=1)
        cols = np.any(body_mask, axis=0)
        if not np.any(rows):
            results.append((
                np.full((crop_h, crop_w, 3), 255, dtype=np.uint8),
                np.zeros((crop_h, crop_w), dtype=bool),
            ))
            continue

        # Center crop on body centroid
        r_indices = np.where(rows)[0]
        c_indices = np.where(cols)[0]
        rc = (r_indices[0] + r_indices[-1]) // 2
        cc = (c_indices[0] + c_indices[-1]) // 2

        r0 = max(0, rc - crop_half_h)
        r1 = min(render_h, r0 + crop_h)
        r0 = r1 - crop_h  # ensure exact size
        c0 = max(0, cc - crop_half_w)
        c1 = min(render_w, c0 + crop_w)
        c0 = c1 - crop_w

        results.append((
            rgb[r0:r1, c0:c1],
            body_mask[r0:r1, c0:c1],
        ))

    rgb_renderer.close()
    seg_renderer.close()
    return results


def _tile_poses(
    cropped_poses: list[tuple[np.ndarray, np.ndarray]],
    tint_color: tuple[float, float, float] = (0.45, 0.60, 0.75),
    gap_fraction: float = -0.15,
) -> np.ndarray:
    """Composite cropped poses into a horizontal filmstrip with tinting."""
    if not cropped_poses:
        return np.full((100, 200, 3), 255, dtype=np.uint8)

    n = len(cropped_poses)
    tint = np.array(tint_color)
    tint_start = np.clip(tint * 0.5 + 0.5, 0, 1)
    tint_end = np.clip(tint * 0.75, 0, 1)

    max_h = max(c[0].shape[0] for c in cropped_poses)

    normalized = []
    for crop_rgb, crop_mask in cropped_poses:
        h, w = crop_rgb.shape[:2]
        if h < max_h:
            pt = (max_h - h) // 2
            pb = max_h - h - pt
            crop_rgb = np.pad(
                crop_rgb, ((pt, pb), (0, 0), (0, 0)),
                mode="constant", constant_values=255,
            )
            crop_mask = np.pad(
                crop_mask, ((pt, pb), (0, 0)),
                mode="constant", constant_values=False,
            )
        normalized.append((crop_rgb, crop_mask))

    avg_w = sum(c[0].shape[1] for c in normalized) // n
    gap = max(int(avg_w * gap_fraction), 2)
    total_w = sum(c[0].shape[1] for c in normalized) + gap * (n - 1)

    composite = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for i, (crop_rgb, crop_mask) in enumerate(normalized):
        h, w = crop_rgb.shape[:2]
        t = i / max(n - 1, 1)
        tint_t = tint_start * (1 - t) + tint_end * t

        if np.any(crop_mask):
            body = crop_rgb[crop_mask].astype(np.float32) / 255.0
            lum = 1.0 - np.mean(body, axis=-1, keepdims=True)
            colored = lum * tint_t[None, :] + (1.0 - lum)
            tinted = np.full_like(crop_rgb, 255)
            tinted[crop_mask] = np.clip(colored * 255.0, 0, 255).astype(np.uint8)
        else:
            tinted = crop_rgb

        composite[:h, x : x + w] = tinted
        x += w + gap

    return composite[:, : max(1, x - gap)]


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    oh, ow = img.shape[:2]
    if oh == 0 or ow == 0:
        return np.full((h, h, 3), 255, dtype=np.uint8)
    nw = max(1, int(ow * h / oh))
    return np.array(Image.fromarray(img).resize((nw, h), Image.LANCZOS))


def render_syllable_filmstrip(
    qpos_clip: np.ndarray,
    start: int,
    end: int,
    xml_path: str,
    n_poses: int = 3,
    tint_color: tuple[float, float, float] = (0.45, 0.60, 0.75),
    fixed_crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Render *n_poses* evenly sampled from frames [start, end)."""
    frames = list(range(start, end))
    if len(frames) > n_poses:
        idx = np.linspace(0, len(frames) - 1, n_poses).round().astype(int)
        frames = [frames[int(i)] for i in idx]

    cropped = _render_cropped_poses(qpos_clip, frames, xml_path, fixed_crop=fixed_crop)
    return _tile_poses(cropped, tint_color=tint_color)


# ── Figure assembly ──────────────────────────────────────────────────────────

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
}


def _replace_white_bg(
    img: np.ndarray, color: tuple[int, int, int], thresh: int = 248
) -> np.ndarray:
    out = img.copy()
    out[np.all(img >= thresh, axis=-1)] = color
    return out


def make_clip_figure(
    clip_codes: np.ndarray,
    qpos_clip: np.ndarray,
    clip_idx: int,
    output_path: str,
    xml_path: str | None = None,
    max_poses: int = 3,
    min_seg_frames: int = 5,
    fixed_crop: tuple[int, int, int, int] | None = None,
) -> None:
    """Create one horizontal figure for a single clip.

    Top row: colored code bar proportional to syllable durations.
    Bottom row: filmstrip per syllable aligned to the segment width,
    with straight-down arrows. Narrow segments get fewer poses (2 or 1).
    """
    if xml_path is None:
        xml_path = _get_xml_path()

    segments = find_syllable_segments(clip_codes, min_frames=min_seg_frames)
    if len(segments) < 2:
        print(f"  Clip {clip_idx}: only {len(segments)} syllable(s), skipping")
        return
    # Cap at 5 syllables for readability
    segments = segments[:5]

    color_map = _code_color_map(np.array([c for c, _, _ in segments]))
    n_segs = len(segments)
    total_frames = sum(e - s for _, s, e in segments)

    # Decide number of poses per segment based on relative width
    seg_fracs = [(e - s) / total_frames for _, s, e in segments]
    median_frac = float(np.median(seg_fracs))
    poses_per_seg: list[int] = []
    for frac in seg_fracs:
        ratio = frac / median_frac
        if ratio < 0.35:
            poses_per_seg.append(1)
        elif ratio < 0.65:
            poses_per_seg.append(2)
        else:
            poses_per_seg.append(min(max_poses, 3))

    # Render filmstrips (per-segment pose count)
    print(f"  Clip {clip_idx}: rendering {n_segs} syllables "
          f"(poses: {poses_per_seg}) ...")
    strips: list[np.ndarray] = []
    for (code, s, e), np_ in zip(segments, poses_per_seg):
        rgb = matplotlib.colors.to_rgb(color_map[code])
        strip = render_syllable_filmstrip(
            qpos_clip, s, e, xml_path, n_poses=np_, tint_color=rgb,
            fixed_crop=fixed_crop,
        )
        strips.append(strip)

    # Figure sizing — give enough horizontal room
    fig_w = max(14.0, n_segs * 2.5 + 1.0)
    fig_h = 5.5

    with plt.rc_context(_NATURE_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")

        bar_top = 0.93
        bar_bot = 0.83
        strip_top = 0.76
        strip_bot = 0.03
        margin_lr = 0.03
        avail_w = 1.0 - 2 * margin_lr

        # --- Draw code bar (proportional widths) ---
        x_cursor = margin_lr
        seg_lefts: list[float] = []
        seg_rights: list[float] = []
        seg_mids: list[float] = []

        for code, s, e in segments:
            frac = (e - s) / total_frames
            seg_w = frac * avail_w

            fig.patches.append(FancyBboxPatch(
                (x_cursor, bar_bot), seg_w, bar_top - bar_bot,
                boxstyle="round,pad=0.003",
                facecolor=color_map[code], edgecolor="#444444",
                linewidth=0.6, zorder=3,
                transform=fig.transFigure, clip_on=False,
            ))
            fig.text(
                x_cursor + seg_w / 2, (bar_top + bar_bot) / 2,
                f"$c_{{{code}}}$", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4,
            )

            seg_lefts.append(x_cursor)
            seg_rights.append(x_cursor + seg_w)
            seg_mids.append(x_cursor + seg_w / 2)
            x_cursor += seg_w

        # --- Draw filmstrips aligned to segment widths ---
        dy_fig = strip_top - strip_bot
        pad_x = 0.004  # tiny inset from segment edges

        for i, (strip, (code, s, e)) in enumerate(zip(strips, segments)):
            seg_w = seg_rights[i] - seg_lefts[i]
            ax_x = seg_lefts[i] + pad_x
            ax_w = seg_w - 2 * pad_x

            # Straight-down arrow from segment center
            fig.patches.append(FancyArrowPatch(
                (seg_mids[i], bar_bot - 0.01),
                (seg_mids[i], strip_top + 0.01),
                arrowstyle="-|>", color="#666666", lw=0.8,
                mutation_scale=8, zorder=2,
                transform=fig.transFigure, clip_on=False,
            ))

            # Filmstrip axes — match segment width, anchor to top
            # Compute height from image aspect to avoid whitespace
            img_h, img_w = strip.shape[:2]
            ax_h_from_aspect = ax_w * (fig_w / fig_h) * (img_h / img_w)
            ax_h = min(dy_fig, ax_h_from_aspect)
            ax_y = strip_top - ax_h  # anchor to top (close to arrows)
            ax = fig.add_axes([ax_x, ax_y, ax_w, ax_h])
            ax.imshow(strip)
            ax.set_axis_off()

        fig.suptitle(
            f"Clip {clip_idx}  —  KPMS Syllable Behaviors  "
            f"({n_segs} syllables, "
            f"{len(set(c for c, _, _ in segments))} unique codes)",
            fontsize=9, fontweight="bold", y=0.99,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        base = str(output_path).replace(".png", "").replace(".pdf", "")
        for ext in (".pdf", ".png"):
            fig.savefig(base + ext, dpi=400, bbox_inches="tight",
                        facecolor="white", pad_inches=0.04)
        plt.close(fig)
    print(f"  Saved: {base}.pdf / .png")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Render KPMS syllable filmstrips for selected clips",
    )
    parser.add_argument(
        "--codes", default=DEFAULT_CODES,
        help="Path to best_codes.npz from KPMS sweep",
    )
    parser.add_argument(
        "--h5", default=DEFAULT_H5,
        help="Path to rodent reference clips .h5",
    )
    parser.add_argument(
        "--clips", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Clip indices to visualize (from all_codes)",
    )
    parser.add_argument(
        "--output", default="outputs/kpms_filmstrips",
        help="Output directory",
    )
    parser.add_argument("--max-poses", type=int, default=3)
    parser.add_argument("--min-seg-frames", type=int, default=5)
    args = parser.parse_args()

    codes_data = np.load(args.codes)
    all_codes = codes_data["all_codes"]
    n_clips_total, n_frames = all_codes.shape
    print(f"Loaded codes: {all_codes.shape} ({n_clips_total} clips, {n_frames} frames)")

    with h5py.File(args.h5, "r") as f:
        qpos_all = f["qpos"][:]  # (total_frames, 74)
    frames_per_clip = n_frames
    print(f"Loaded qpos: {qpos_all.shape}")

    xml_path = _get_xml_path()
    print(f"XML: {xml_path}")

    # Auto-select clips with the most syllables if none specified
    clip_list = args.clips
    if clip_list is None:
        scored = []
        for ci in range(n_clips_total):
            segs = find_syllable_segments(all_codes[ci], min_frames=args.min_seg_frames)
            if len(segs) >= 5:
                scored.append((ci, len(segs)))
        scored.sort(key=lambda x: -x[1])
        clip_list = [ci for ci, _ in scored[:8]]
        print(f"Auto-selected {len(clip_list)} clips with >= 5 syllables "
              f"(min {args.min_seg_frames} frames): {clip_list}")

    # Pre-pass: compute a global crop SIZE so all rodents render at the same scale.
    # Uses p95 of body widths/heights (not full union which is too large).
    print("Computing global crop size across all clips ...")
    body_widths: list[int] = []
    body_heights: list[int] = []
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    seg_renderer = mujoco.Renderer(mj_model, height=800, width=800)
    seg_renderer.enable_segmentation_rendering()
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    fovy_rad = math.radians(45.0)
    view_h = 0.35
    dist = (view_h / 2.0) / math.tan(fovy_rad / 2.0) * 1.3
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = dist
    camera.elevation = -35.0
    camera.azimuth = 90.0

    for clip_idx in clip_list:
        if clip_idx >= n_clips_total:
            continue
        qpos_clip = qpos_all[clip_idx * frames_per_clip : (clip_idx + 1) * frames_per_clip]
        sample_frames = np.linspace(0, min(qpos_clip.shape[0] - 1, frames_per_clip - 1), 10).round().astype(int)
        for fi in sample_frames:
            camera.lookat[:] = [qpos_clip[fi, 0], qpos_clip[fi, 1], qpos_clip[fi, 2] + 0.02]
            mj_data.qpos[:] = qpos_clip[fi]
            mujoco.mj_forward(mj_model, mj_data)
            seg_renderer.update_scene(mj_data, camera=camera)
            seg = seg_renderer.render()
            bm = (seg[:, :, 0].astype(np.int32) >= 0) & (seg[:, :, 0].astype(np.int32) != floor_id)
            rows = np.any(bm, axis=1)
            cols = np.any(bm, axis=0)
            if np.any(rows):
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                body_widths.append(c1 - c0)
                body_heights.append(r1 - r0)

    seg_renderer.close()
    # Use p95 + generous padding for uniform crop size
    pad = 20
    crop_half_h = int(np.percentile(body_heights, 95) / 2) + pad
    crop_half_w = int(np.percentile(body_widths, 95) / 2) + pad
    global_crop = (0, 2 * crop_half_h, 0, 2 * crop_half_w)
    print(f"Global crop size: {2*crop_half_h} x {2*crop_half_w} px "
          f"(p95 body: {np.percentile(body_heights,95):.0f}h x {np.percentile(body_widths,95):.0f}w)")

    for clip_idx in clip_list:
        if clip_idx >= n_clips_total:
            print(f"  Clip {clip_idx} out of range, skipping")
            continue

        clip_codes = all_codes[clip_idx]
        # Slice qpos for this clip
        qpos_clip = qpos_all[clip_idx * frames_per_clip : (clip_idx + 1) * frames_per_clip]
        if qpos_clip.shape[0] == 0:
            print(f"  Clip {clip_idx}: no qpos data, skipping")
            continue

        out_path = os.path.join(args.output, f"clip_{clip_idx}_syllables.png")
        make_clip_figure(
            clip_codes, qpos_clip, clip_idx, out_path,
            xml_path=xml_path,
            max_poses=args.max_poses,
            min_seg_frames=args.min_seg_frames,
            fixed_crop=global_crop,
        )


if __name__ == "__main__":
    main()
