"""Rendering utilities for rollout visualization and analysis.

This module provides functions for rendering MuJoCo rollouts with optional
PCA progression overlays, and displaying videos in Jupyter notebooks.

Note:
    Sets MUJOCO_GL and PYOPENGL_PLATFORM environment variables to "egl"
    for headless rendering if not already set.
"""

import functools
import multiprocessing as mp
import os
from typing import Any

# Configure OpenGL for headless rendering (must be before matplotlib import)
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm


def plot_pca_intention(
    idx: int,
    episode_start: int,
    pca: PCA,
    pca_projections: np.ndarray,
    clip_idx: int,
    feature_name: str,
    n_components: int = 4,
    terminated: bool = False,
    window_size: int = 530,
) -> np.ndarray:
    """Generate a PCA progression plot as an image array.

    Creates a line plot showing the trajectory of PCA components over time,
    with the current timestep marked. Useful for visualizing latent intention
    or control signal evolution during a rollout.

    Args:
        idx: Current timestep index (absolute, not relative to episode).
        episode_start: Timestep where the current episode began.
        pca: Fitted sklearn PCA object (used for variance ratios in legend).
        pca_projections: PCA-transformed data, shape (num_timesteps, n_components).
        clip_idx: Reference clip index (displayed in title).
        feature_name: Name of the feature being visualized (e.g., "ctrl", "intention").
        n_components: Number of principal components to plot. Defaults to 4.
        terminated: If True, draws a vertical line marking episode termination.
        window_size: Number of timesteps visible in the x-axis window. Defaults to 530.

    Returns:
        RGB image array of the plot, shape (height, width, 3).
    """
    max_y = np.max(pca_projections[:, :n_components])
    min_y = np.min(pca_projections[:, :n_components])
    y_lim = (min_y - 0.2, max_y + 0.2)
    idx_in_episode = idx - episode_start

    plt.figure(figsize=(9.6, 4.8))

    for pc_ind in range(n_components):
        variance_pct = pca.explained_variance_ratio_[pc_ind] * 100
        plt.plot(
            pca_projections[episode_start:idx, pc_ind],
            label=f"PC {pc_ind} ({variance_pct:.1f}%)",
        )
        plt.scatter(idx_in_episode, pca_projections[idx - 1, pc_ind])

    if terminated:
        plt.axvline(x=idx_in_episode, color="r", linestyle="-")
        plt.text(
            idx_in_episode - 8,
            sum(y_lim) / 2,
            "Episode Terminated",
            color="r",
            rotation=90,
        )

    # Sliding window for x-axis
    if idx_in_episode <= window_size:
        plt.xlim(0, window_size)
    else:
        plt.xlim(idx_in_episode - window_size, idx_in_episode)

    plt.ylim(*y_lim)
    plt.legend(loc="upper right")
    plt.xlabel("Timestep")
    plt.title(f"PCA {feature_name} Progression for Clip {clip_idx}")

    # Render figure to numpy array
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf, (width, height) = canvas.print_to_buffer()
    image = Image.frombytes("RGBA", (width, height), buf)
    rgb_array = np.array(image.convert("RGB"))
    plt.close(fig)

    return rgb_array


def render_with_pca_progression(
    rollout: dict[str, Any],
    pca: PCA,
    pca_projections: np.ndarray,
    render_fn: callable,
    n_components: int = 4,
    feature_name: str = "ctrl",
) -> list[np.ndarray]:
    """Render rollout frames side-by-side with PCA progression plots.

    Combines MuJoCo rendered frames with corresponding PCA trajectory plots,
    creating a visualization that shows both the physical simulation and
    the evolution of latent features over time.

    Args:
        rollout: Rollout dictionary containing "info" and "qposes_rollout" keys.
        pca: Fitted sklearn PCA object for variance ratio display.
        pca_projections: PCA-transformed features, shape (num_timesteps, n_components).
        render_fn: Function to render rollout states, returns list of frame arrays.
        n_components: Number of PCA components to display. Defaults to 4.
        feature_name: Label for the PCA plot title. Defaults to "ctrl".

    Returns:
        List of concatenated frames (MuJoCo frame | PCA plot), suitable for video.

    Note:
        Adds 50 frozen frames at the end showing the termination state.
    """
    # Skip first frame (no intention data available)
    frames_mujoco = render_fn(rollout)[1:]

    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    clip_idx = int(rollout["info"][0]["clip_idx"])

    # Parallel render PCA plots
    worker = functools.partial(
        plot_pca_intention,
        episode_start=0,
        pca=pca,
        pca_projections=pca_projections,
        clip_idx=clip_idx,
        n_components=n_components,
        feature_name=feature_name,
    )

    print("Rendering PCA progression plots...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_pca = pool.map(worker, range(len(rollout["qposes_rollout"])))

    print("Concatenating frames...")
    concat_frames = []
    for idx, frame in tqdm(enumerate(frames_mujoco)):
        concat_frames.append(np.hstack([frame, frames_pca[idx]]))

    # Final frame with termination marker
    final_plot = plot_pca_intention(
        idx=len(frames_mujoco) - 1,
        episode_start=0,
        pca=pca,
        pca_projections=pca_projections,
        clip_idx=clip_idx,
        feature_name=feature_name,
        n_components=n_components,
        terminated=True,
    )

    plt.close("all")
    matplotlib.use(orig_backend)

    # Add frozen end frames
    for _ in range(50):
        concat_frames.append(np.hstack([frames_mujoco[-1], final_plot]))

    return concat_frames


def display_video(frames: list[np.ndarray], framerate: int = 30) -> HTML:
    """Display a sequence of frames as an HTML5 video in Jupyter.

    Creates a matplotlib animation from the frames and converts it to an
    HTML5 video element for inline display in Jupyter notebooks.

    Args:
        frames: List of RGB image arrays, each with shape (height, width, 3).
        framerate: Video playback speed in frames per second. Defaults to 30.

    Returns:
        IPython HTML object containing the embedded video.

    Example:
        >>> frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ...           for _ in range(100)]
        >>> display_video(frames, framerate=30)
    """
    height, width, _ = frames[0].shape
    dpi = 70

    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    plt.close("all")
    matplotlib.use(orig_backend)

    def update(frame: np.ndarray) -> list[Any]:
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=frames,
        interval=interval,
        blit=True,
        repeat=False,
    )

    return HTML(anim.to_html5_video())
