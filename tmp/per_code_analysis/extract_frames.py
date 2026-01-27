#!/usr/bin/env python3
"""Extract frame montages from per-code videos for analysis."""

import sys
sys.path.insert(0, '/home/jovyan/vast/kaiwen/track-mjx')

import imageio
import numpy as np
from PIL import Image
from pathlib import Path

VIDEO_DIR = Path("/home/jovyan/vast/kaiwen/track-mjx/vqvae_jax/wandb/run-20260125_202952-vqvae_multi_clip_260125_202945_718947/files/media/videos/videos/per_code")
OUTPUT_DIR = Path("/home/jovyan/vast/kaiwen/track-mjx/tmp/per_code_analysis")

def extract_montage(video_path: Path, n_frames: int = 6) -> np.ndarray:
    """Extract evenly spaced frames and create a horizontal montage."""
    reader = imageio.get_reader(video_path)
    frames = list(reader)
    reader.close()

    if len(frames) == 0:
        return None

    # Sample frames evenly
    indices = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)), dtype=int)
    sampled = [frames[i] for i in indices]

    # Resize frames to consistent height
    target_height = 200
    resized = []
    for f in sampled:
        img = Image.fromarray(f)
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        resized.append(np.array(img))

    # Create horizontal montage
    montage = np.concatenate(resized, axis=1)
    return montage

def main():
    codes_to_analyze = [3, 5, 11, 14]  # Codes that appear across multiple checkpoints

    for code in codes_to_analyze:
        print(f"\n=== Code {code} ===")
        video_files = sorted(VIDEO_DIR.glob(f"code_{code}_*.mp4"))

        if not video_files:
            print(f"  No videos found for code {code}")
            continue

        montages = []
        labels = []

        for vf in video_files:
            # Extract checkpoint from filename: code_X_CHECKPOINT_hash.mp4
            parts = vf.stem.split('_')
            checkpoint = parts[2]

            try:
                montage = extract_montage(vf, n_frames=6)
                if montage is not None:
                    montages.append(montage)
                    labels.append(f"ckpt {checkpoint}")
                    print(f"  Checkpoint {checkpoint}: {len(list(imageio.get_reader(vf)))} frames")
            except Exception as e:
                print(f"  Error processing {vf.name}: {e}")

        if montages:
            # Pad montages to same width
            max_width = max(m.shape[1] for m in montages)
            padded = []
            for m in montages:
                if m.shape[1] < max_width:
                    pad = np.ones((m.shape[0], max_width - m.shape[1], 3), dtype=np.uint8) * 255
                    m = np.concatenate([m, pad], axis=1)
                padded.append(m)

            # Stack vertically
            combined = np.concatenate(padded, axis=0)

            # Save
            output_path = OUTPUT_DIR / f"code_{code}_comparison.png"
            Image.fromarray(combined).save(output_path)
            print(f"  Saved to {output_path}")

if __name__ == "__main__":
    main()
