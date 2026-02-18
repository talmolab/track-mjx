"""Preprocess reference clips into balanced category splits.

Classifies 842 rodent reference clips into 4 movement categories based on
kinematic statistics (XY displacement and Z range), balances categories by
subsampling to the minimum count, and creates stratified train/test splits.

Usage:
    cd vqvae_jax
    python preprocess_clips.py --data_path /path/to/rodent_reference_clips.h5

Output:
    data/rodent/rodent_balanced_splits.json
    data/rodent/balanced_preview_grid.mp4
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_clip_metrics(
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-clip XY displacement and Z range from qpos.

    Args:
        qpos: Reference clip positions, shape [N_clips, T, n_qpos].
            Root position is qpos[:, :, 0:3] (x, y, z).

    Returns:
        xy_displacement: Per-clip total XY path length, shape [N_clips].
        z_range: Per-clip Z range (max - min), shape [N_clips].
    """
    root_xy = qpos[:, :, 0:2]  # [N, T, 2]
    root_z = qpos[:, :, 2]  # [N, T]

    # XY path length: sum of frame-to-frame distances
    diffs = np.diff(root_xy, axis=1)  # [N, T-1, 2]
    frame_distances = np.linalg.norm(diffs, axis=-1)  # [N, T-1]
    xy_displacement = np.sum(frame_distances, axis=-1)  # [N]

    # Z range: max - min
    z_range = np.max(root_z, axis=-1) - np.min(root_z, axis=-1)  # [N]

    return xy_displacement, z_range


def classify_clips(
    xy_displacement: np.ndarray,
    z_range: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Classify clips into 4 categories using median thresholds.

    Categories:
        groom: low XY, low Z (stationary grooming)
        rear: low XY, high Z (rearing up)
        walk: high XY, low Z (horizontal locomotion)
        rear_walk: high XY, high Z (locomotion with rearing)

    Args:
        xy_displacement: Per-clip XY path length, shape [N].
        z_range: Per-clip Z range, shape [N].

    Returns:
        categories: Dict mapping category name to array of clip indices.
        xy_threshold: Median XY threshold used.
        z_threshold: Median Z threshold used.
    """
    xy_threshold = float(np.median(xy_displacement))
    z_threshold = float(np.median(z_range))

    low_xy = xy_displacement < xy_threshold
    high_xy = ~low_xy
    low_z = z_range < z_threshold
    high_z = ~low_z

    categories = {
        "groom": np.where(low_xy & low_z)[0],
        "rear": np.where(low_xy & high_z)[0],
        "walk": np.where(high_xy & low_z)[0],
        "rear_walk": np.where(high_xy & high_z)[0],
    }

    return categories, xy_threshold, z_threshold


def balance_and_split(
    categories: dict[str, np.ndarray],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Balance categories and create stratified train/test split.

    Subsamples each category to the minimum category count, then splits
    each category into train/test according to train_ratio.

    Args:
        categories: Dict mapping category name to array of clip indices.
        train_ratio: Fraction of each category to use for training.
        seed: Random seed for reproducibility.

    Returns:
        train_indices: List of clip indices for training.
        test_indices: List of clip indices for testing.
        train_categories: Category label for each train index.
        test_categories: Category label for each test index.
    """
    rng = np.random.RandomState(seed)

    # Find minimum category count
    min_count = min(len(indices) for indices in categories.values())
    logging.info(
        f"Category sizes: {
        {k: len(v) for k, v in categories.items()}
    }"
    )
    logging.info(f"Balancing to min count: {min_count}")

    train_indices = []
    test_indices = []
    train_categories = []
    test_categories = []

    for cat_name, cat_indices in categories.items():
        # Subsample to min_count
        selected = rng.choice(cat_indices, size=min_count, replace=False)
        selected = rng.permutation(selected)

        # Split
        n_train = int(len(selected) * train_ratio)
        train = selected[:n_train].tolist()
        test = selected[n_train:].tolist()

        train_indices.extend(train)
        test_indices.extend(test)
        train_categories.extend([cat_name] * len(train))
        test_categories.extend([cat_name] * len(test))

    return train_indices, test_indices, train_categories, test_categories


def render_category_grid(
    qpos: np.ndarray,
    categories: dict[str, np.ndarray],
    xy_displacement: np.ndarray,
    z_range: np.ndarray,
    data_path: str,
    output_path: str | Path,
    fps: int = 50,
    width: int = 320,
    height: int = 240,
    camera: str = "close_profile-rodent",
    frame_step: int = 1,
) -> str:
    """Render a 2x2 grid video with one representative clip per category.

    Picks the clip closest to the median XY/Z values within each category
    for a representative sample. Renders reference qpos directly (no policy).

    Args:
        qpos: All clips qpos, shape [N, T, nq].
        categories: Dict mapping category name to clip index arrays.
        xy_displacement: Per-clip XY path lengths, shape [N].
        z_range: Per-clip Z ranges, shape [N].
        data_path: Path to the reference clips H5 file.
        output_path: Path for output MP4 file.
        fps: Video frame rate.
        width: Per-panel width in pixels.
        height: Per-panel height in pixels.
        camera: MuJoCo camera name.
        frame_step: Render every N-th frame (1 = all frames).

    Returns:
        Output path as string.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from vnl_playground.tasks.rodent import imitation

    # Build MuJoCo model via the Imitation environment (handles rescaling)
    from etils.epath.gpath import PosixGPath

    cfg = imitation.default_config()
    cfg.reference_data_path = PosixGPath(data_path)
    single_clip = ReferenceClips(
        data_path=data_path,
        n_frames_per_clip=qpos.shape[1],
        keep_clips_idx=np.array([0]),
    )
    env = imitation.Imitation(config=cfg, clips=single_clip)
    model = env.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Pick one representative clip per category (closest to category median)
    cat_names = ["groom", "rear", "walk", "rear_walk"]
    cat_labels = {
        "groom": "Groom (low XY, low Z)",
        "rear": "Rear (low XY, high Z)",
        "walk": "Walk (high XY, low Z)",
        "rear_walk": "Rear+Walk (high XY, high Z)",
    }
    representative_clips = {}
    for cat in cat_names:
        idx = categories[cat]
        if len(idx) == 0:
            logging.warning(f"Category '{cat}' is empty, skipping")
            continue
        # Pick clip closest to category's median XY and Z
        med_xy = np.median(xy_displacement[idx])
        med_z = np.median(z_range[idx])
        dist = (xy_displacement[idx] - med_xy) ** 2 + (z_range[idx] - med_z) ** 2
        representative_clips[cat] = idx[np.argmin(dist)]

    logging.info(
        f"Representative clips: {
            {k: int(v) for k, v in representative_clips.items()}
        }"
    )

    # Render frames for each representative clip
    n_frames = qpos.shape[1]
    frame_indices = range(0, n_frames, frame_step)

    # Try to load a font for labels
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    grid_frames = []
    for t in frame_indices:
        panels = {}
        for cat in cat_names:
            if cat not in representative_clips:
                panels[cat] = np.zeros((height, width, 3), dtype=np.uint8)
                continue
            clip_idx = representative_clips[cat]
            data.qpos[:] = np.array(qpos[clip_idx, t, :])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()

            # Add label overlay
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            label = cat_labels[cat]
            # Draw text with dark background for readability
            draw.rectangle([(0, 0), (width, 20)], fill=(0, 0, 0, 180))
            draw.text((4, 3), label, fill=(255, 255, 255), font=font)
            panels[cat] = np.array(img)

        # Assemble 2x2 grid: [groom, rear] / [walk, rear_walk]
        top = np.concatenate([panels["groom"], panels["rear"]], axis=1)
        bottom = np.concatenate([panels["walk"], panels["rear_walk"]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        grid_frames.append(grid)

    renderer.close()

    # Write video
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in grid_frames:
        writer.append_data(frame)
    writer.close()

    logging.info(
        f"Saved grid preview video ({len(grid_frames)} frames) to: {output_path}"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess clips into balanced splits"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to rodent reference clips H5 file. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=250,
        help="Number of frames per clip (default: 250).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of balanced clips for training (default: 0.7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path. Default: data/rodent/rodent_balanced_splits.json",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip rendering the preview grid video.",
    )
    args = parser.parse_args()

    # Auto-detect data path
    if args.data_path is None:
        # Try common locations
        candidates = [
            Path(
                "/home/jovyan/vast/kaiwen/track-mjx/data/rodent/rodent_reference_clips.h5"
            ),
        ]
        for p in candidates:
            if p.exists():
                args.data_path = str(p)
                break
        if args.data_path is None:
            # Try using vnl_playground's default path
            try:
                from vnl_playground.tasks.rodent import imitation

                default_cfg = imitation.ImitationConfig()
                args.data_path = default_cfg.reference_data_path
                logging.info(f"Using vnl_playground default: {args.data_path}")
            except Exception:
                raise ValueError(
                    "Could not auto-detect data path. Please specify --data_path."
                )

    logging.info(f"Loading reference clips from: {args.data_path}")

    # Load all clips
    clips = ReferenceClips(
        data_path=args.data_path,
        n_frames_per_clip=args.clip_length,
    )

    # Extract qpos
    qpos = np.array(clips.qpos)  # [N, T, n_qpos]
    n_clips = qpos.shape[0]
    logging.info(f"Loaded {n_clips} clips, qpos shape: {qpos.shape}")

    # Compute metrics
    xy_displacement, z_range = compute_clip_metrics(qpos)
    logging.info(
        f"XY displacement: min={xy_displacement.min():.4f}, "
        f"max={xy_displacement.max():.4f}, median={np.median(xy_displacement):.4f}"
    )
    logging.info(
        f"Z range: min={z_range.min():.4f}, "
        f"max={z_range.max():.4f}, median={np.median(z_range):.4f}"
    )

    # Classify
    categories, xy_threshold, z_threshold = classify_clips(xy_displacement, z_range)
    logging.info(f"Thresholds: XY={xy_threshold:.4f}, Z={z_threshold:.4f}")
    for cat_name, cat_indices in categories.items():
        logging.info(f"  {cat_name}: {len(cat_indices)} clips")

    # Balance and split
    train_indices, test_indices, train_cats, test_cats = balance_and_split(
        categories, train_ratio=args.train_ratio, seed=args.seed
    )
    logging.info(
        f"Balanced split: {len(train_indices)} train, {len(test_indices)} test"
    )

    # Build output JSON
    output = {
        "metadata": {
            "source_path": args.data_path,
            "n_clips_total": n_clips,
            "n_clips_balanced": len(train_indices) + len(test_indices),
            "xy_threshold": xy_threshold,
            "z_threshold": z_threshold,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "clip_length": args.clip_length,
            "created": datetime.now().isoformat(),
        },
        "per_clip_stats": {
            "xy_displacement": xy_displacement.tolist(),
            "z_range": z_range.tolist(),
        },
        "categories": {k: v.tolist() for k, v in categories.items()},
        "balanced": {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_categories": train_cats,
            "test_categories": test_cats,
        },
    }

    # Write output
    if args.output_path is None:
        output_dir = Path(args.data_path).parent
        output_path = output_dir / "rodent_balanced_splits.json"
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logging.info(f"Saved balanced splits to: {output_path}")

    # Render preview grid video
    if not args.no_video:
        video_path = output_path.parent / "balanced_preview_grid.mp4"
        try:
            render_category_grid(
                qpos=qpos,
                categories=categories,
                xy_displacement=xy_displacement,
                z_range=z_range,
                data_path=args.data_path,
                output_path=video_path,
            )
        except Exception as e:
            logging.warning(f"Failed to render preview video: {e}")


if __name__ == "__main__":
    main()
