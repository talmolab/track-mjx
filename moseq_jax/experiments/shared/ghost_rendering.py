"""K-body ghost rendering — thin wrapper around vqvae_jax rendering code.

Re-exports the ghost model construction and video rendering functions from
``vqvae_jax.ablation.run_divergent_futures``, plus helpers for solo-body
rendering and colour generation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import imageio
import mujoco
import numpy as np

# Ensure repo root importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vqvae_jax.ablation.run_divergent_futures import (
    build_ghost_model,
    render_ghost_video,
    _disable_lights_recursive,
    _make_code_bar,
)
from vqvae_jax.analysis.rendering import (
    add_multi_line_overlay,
    get_code_colormap as _vqvae_get_code_colormap,
)

# Re-export for convenience
__all__ = [
    "build_ghost_model",
    "render_ghost_video",
    "_disable_lights_recursive",
    "_make_code_bar",
    "render_solo_video",
]

GHOST_CAMERA = "divergent_cam"


def render_solo_video(
    env: Any,
    rollout_qpos: np.ndarray,
    code_indices: np.ndarray | None,
    output_path: str | Path,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    num_codes: int = 50,
    title: str = "",
) -> str:
    """Render a single-body rollout video with optional code timeline bar.

    Args:
        env: Imitation environment (for ``mj_model``).
        rollout_qpos: ``[T, nq]`` joint configuration trajectory.
        code_indices: ``[T]`` code indices (or ``None`` to skip bar).
        output_path: Output MP4 path.
        camera: Camera name (uses env default if ``None``).
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        num_codes: Codebook size (for colour map).
        title: Title overlay text.

    Returns:
        Path string to written video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mj_model = env.mj_model
    mj_model.vis.global_.offwidth = width
    mj_model.vis.global_.offheight = height
    data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    if camera is None:
        camera = f"close_profile{getattr(env, '_suffix', '')}"

    code_colors = None
    if code_indices is not None:
        from .plotting import get_code_colormap
        code_colors = get_code_colormap(num_codes)

    T = len(rollout_qpos)
    writer = imageio.get_writer(str(output_path), fps=fps)

    for t in range(T):
        data.qpos[:] = rollout_qpos[t]
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render().copy()

        if title:
            frame = add_multi_line_overlay(
                frame, [title, f"t={t}"], start_position=(10, 10), font_size=14,
            )

        if code_colors is not None and code_indices is not None:
            bar_img = _make_code_bar(
                width=width,
                code_sequences=[code_indices],
                frame_idx=t,
                colors=[(200, 200, 200)],
                code_colors=code_colors,
                bar_height=20,
            )
            frame[-bar_img.shape[0]:, :] = bar_img

        writer.append_data(frame)

    writer.close()
    renderer.close()
    logging.info(f"  Wrote solo video ({T} frames): {output_path}")
    return str(output_path)
