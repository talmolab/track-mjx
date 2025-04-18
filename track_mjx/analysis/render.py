# imports
import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
from IPython.display import HTML

from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.walker.rodent import Rodent

import mujoco
from pathlib import Path
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale
import imageio
import numpy as np

import multiprocessing as mp
import functools

from ripser import ripser
from persim import plot_diagrams
from sklearn.manifold import Isomap
from sklearn.neighbors import LocalOutlierFactor

from track_mjx.analysis.utils import subsample_data

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def agg_backend_context(func):
    """
    Decorator to switch to a headless backend during function execution.
    """

    def wrapper(*args, **kwargs):
        orig_backend = matplotlib.get_backend()
        matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
        # Code to execute BEFORE the original function
        result = func(*args, **kwargs)
        # Code to execute AFTER the original function
        plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
        matplotlib.use(orig_backend)
        return result

    return wrapper


def render_from_saved_rollout(
    rollout: dict,
    walker_name: str,
) -> list:
    """
    Render a rollout from saved qposes.

    Args:
        rollout (dict): A dictionary containing the qposes of the reference and rollout trajectories.

    Returns:
        list: list of frames of the rendering
    """
    qposes_ref, qposes_rollout = rollout["qposes_ref"], rollout["qposes_rollout"]
    
    if walker_name == "rodent":
        pair_render_xml_path = str(
        (
            Path(__file__).parent
            / ".."
            / "environment"
            / "walker"
            / "assets"
            / "rodent"
            / "rodent_ghostpair_scale080.xml"
        ).resolve()
        )
        camera_name = "close_profile"
        
        spec = mujoco.MjSpec()
        spec = spec.from_file(str(pair_render_xml_path))
        
        # in training scaled by this amount as well
        for geom in spec.geoms:
            if geom.size is not None:
                geom.size *= 0.95
            if geom.pos is not None:
                geom.pos *= 0.95
    else:
        pair_render_xml_path = str(
        (
            Path(__file__).parent
            / ".."
            / "environment"
            / "walker"
            / "assets"
            / "fruitfly"
            / "fruitfly_force_pair.xml"
        ).resolve()
        )
        camera_name = "track1-0"
        
        spec = mujoco.MjSpec()
        spec = spec.from_file(str(pair_render_xml_path))
        
        # in training scaled by this amount as well
        for geom in spec.geoms:
            if geom.size is not None:
                geom.size *= 1
            if geom.pos is not None:
                geom.pos *= 1

    mj_model = spec.compile()

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]

    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)

    site_id = [
        mj_model.site(i).id
        for i in range(mj_model.nsite)
        if "-0" in mj_model.site(i).name
    ]
    for id in site_id:
        mj_model.site(id).rgba = [1, 0, 0, 1]
    
    for i in range(mj_model.ngeom):
        geom_name = mj_model.geom(i).name
        if "-1" in geom_name:  # ghost
            mj_model.geom(i).rgba = [
                1,
                1,
                1,
                0.5,
            ]  # White color, 50% transparent

    # visual mujoco rendering
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    print("MuJoCo Rendering...")
    for qpos1, qpos2 in tqdm(zip(qposes_rollout, qposes_ref), total=len(qposes_rollout)):
        mj_data.qpos = np.append(qpos1, qpos2)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(
            mj_data,
            camera=camera_name
        )
        pixels = renderer.render()
        frames.append(pixels)
        
    return frames


def plot_pca_intention(
    idx,
    episode_start,
    pca: PCA,
    pca_projections: np.ndarray,
    clip_idx: int,
    feature_name: str,
    n_components: int = 4,
    terminated: bool = False,
    window_size: int = 530,
) -> np.ndarray:
    """
    plot pca intention progression of the episode
    Args:
        idx: the current timestep
        episode_start: the start timestep of the episode
        pca: the pca object fitted on the dataset
        pca_projections: the pca projection of the episode, shape (timestep, n_components)
        clip_idx: the clip index
        feature_name: the feature name
        n_components: the number of pca components to plot
        ylim: the y-axis limit
        terminated: whether the episode is terminated
        window_size: the window size of the plot
    Returns:
        np.ndarray: the image array of the plot
    """
    max_y = np.max(list(pca_projections[:, :n_components]))
    min_y = np.min(list(pca_projections[:, :n_components]))
    y_lim = (min_y - 0.2, max_y + 0.2)
    idx_in_this_episode = idx - episode_start  # the current timestep in this episode
    plt.figure(figsize=(9.6, 4.8))
    for pc_ind in range(n_components):
        # Plot the PCA projection of the episode
        plt.plot(
            pca_projections[episode_start:idx, pc_ind],
            label=f"PC {pc_ind} ({pca.explained_variance_ratio_[pc_ind]*100:.1f}%)",
        )
        plt.scatter(idx - episode_start, pca_projections[idx - 1, pc_ind])
    if terminated:
        # Mark the episode termination
        plt.axvline(x=idx - episode_start, color="r", linestyle="-")
        plt.text(
            idx - episode_start - 8,  # Adjust the x-offset as needed
            sum(y_lim) / 2,  # Adjust the y-position as needed
            "Episode Terminated",
            color="r",
            rotation=90,
        )  # Rotate the text vertically
    if idx_in_this_episode <= window_size:
        plt.xlim(0, window_size)
    else:
        plt.xlim(idx_in_this_episode - window_size, idx_in_this_episode)  # dynamically move xlim as time progress
    plt.ylim(*y_lim)
    plt.legend(loc="upper right")
    plt.xlabel("Timestep")
    plt.title(f"PCA {feature_name} Progression for Clip {clip_idx}")  # TODO make it configurable
    # Get the current figure
    fig = plt.gcf()
    # Create a canvas for rendering
    canvas = FigureCanvasAgg(fig)
    # Render the canvas to a buffer
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    # Convert the buffer to a PIL Image
    image = Image.frombytes("RGBA", (width, height), s)
    rgb_array = np.array(image.convert("RGB"))
    return rgb_array


def render_with_pca_progression(
    rollout: dict,
    pca_projections: np.ndarray,
    n_components: int = 4,
    feature_name: str = "ctrl",
) -> list:
    """
    render with the rewards progression graph concat alongside with the rendering.

    Args:
        rollout (dict): the rollout dictionary
        pca_projections (np.ndarray): the pca projections of the rollout
        n_components (int): the number of pca components to plot
        feature_name (str): the feature name

    Returns:
        list: list of frames of the rendering
    """
    frames_mujoco = render_from_saved_rollout(rollout)[1:]
    # skip the first frame, since we don't have intention for the first frame
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    clip_idx = int(rollout["info"][0]["clip_idx"])
    worker = functools.partial(
        plot_pca_intention,
        episode_start=0,
        clip_idx=clip_idx,
        pca_projections=pca_projections,
        n_components=n_components,
        feature_name=feature_name,
    )
    print("Rendering with PCA progression...")
    # Use multiprocessing to parallelize the rendering of the reward graph
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_pca = pool.map(worker, range(len(rollout["qposes_rollout"])))
    concat_frames = []
    episode_start = 0
    # implement reset logics of the reward graph too.
    print("Concatenating frames...")
    for idx, frame in tqdm(enumerate(frames_mujoco)):
        concat_frames.append(np.hstack([frame, frames_pca[idx]]))
    reward_plot = plot_pca_intention(
        len(frames_mujoco) - 1, episode_start, pca_projections, clip_idx, feature_name, n_components, terminated=True
    )
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)
    for _ in range(50):
        concat_frames.append(np.hstack([frames_mujoco[-1], reward_plot]))  # create stoppage when episode terminates
    return concat_frames


def display_video(frames, framerate=30):
    """
    Args:
        frames (array): (n_frames, height, width, 3)
        framerate (int): the framerate of the video
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def global_local_pca_worker(
    frame_idx: int,
    pca_global: np.ndarray,
    cluster_global: np.ndarray,
    global_subset_indices: np.ndarray,
    pca_local: np.ndarray,
    cluster_local: np.ndarray,
    T: int,
    clip_idx: int,
    var_lcoal: np.ndarray,
    var_global: np.ndarray,
) -> np.ndarray:
    """
    The picklable worker function called by Pool.
    """
    is_terminated = (frame_idx == T - 1)
    return plot_global_and_local_pca_intention_with_clusters(
        pca_global=pca_global,
        cluster_global=cluster_global,
        global_subset_indices=global_subset_indices,
        pca_local=pca_local,
        cluster_local=cluster_local,
        frame_idx_local=frame_idx,
        clip_idx=clip_idx,
        var_lcoal=var_lcoal,
        var_global=var_global,
        terminated=is_terminated,
    )
    

def plot_global_and_local_pca_intention_with_clusters(
    pca_global: np.ndarray,
    cluster_global: np.ndarray, 
    global_subset_indices: np.ndarray, 
    pca_local: np.ndarray, 
    cluster_local: np.ndarray,
    frame_idx_local: int,
    clip_idx: int,
    var_lcoal: np.ndarray,
    var_global: np.ndarray,
    terminated: bool = False
):
    """
    Plot side-by-side:
    1) Global PCA (left subplot): entire dataset in gray, this clip in color, highlight current frame.
    2) Local PCA (right subplot): just this clip, highlight current frame.
    
    Returns an RGB image (H x W x 3) as a numpy array.
    """

    # assume the local embedding is T points and we have T frames. frame_idx_local in [0..T).
    T = len(pca_local)
    if frame_idx_local >= T:
        raise ValueError(f"frame_idx_local={frame_idx_local} exceeds local clip length={T}.")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8), dpi=100)
    
    all_x_g = pca_global[:, 0]
    all_y_g = pca_global[:, 1]
    
    # light grey plot
    ax1.scatter(all_x_g, all_y_g, color="lightgray", alpha=0.5, s=10, label="All data (global)")

    # highlight just the points belonging to this clip
    # pca_global_clip shape is (T, 2) if global_subset_indices has length T
    pca_global_clip = pca_global[global_subset_indices]
    cluster_global_clip = cluster_global[global_subset_indices]

    x_clip_g = pca_global_clip[:, 0]
    y_clip_g = pca_global_clip[:, 1]

    sc = ax1.scatter(
        x_clip_g,
        y_clip_g,
        c=cluster_global_clip,
        cmap="tab10",
        alpha=0.7,
        s=30,
        label=f"Clip {clip_idx}"
    )

    # frame_idx_local corresponds to subset_indices[frame_idx_local] in the global space
    current_global_idx = global_subset_indices[frame_idx_local]
    cur_x_g = pca_global[current_global_idx, 0]
    cur_y_g = pca_global[current_global_idx, 1]
    cur_clust_g = cluster_global[current_global_idx]

    ax1.scatter(
        cur_x_g, 
        cur_y_g, 
        c=[cur_clust_g], 
        cmap="tab10",
        edgecolor="black",
        s=100,
        alpha=1.0
    )
    ax1.set_title(f"Global PCA (Clip {clip_idx})")
    ax1.set_xlabel(f"PC1 (global) ({var_global[0]*100:.2f}%)")
    ax1.set_ylabel(f"PC2 (global) ({var_global[1]*100:.2f}%)")
    

    x_l = pca_local[:, 0]
    y_l = pca_local[:, 1]
    sc2 = ax2.scatter(
        x_l, 
        y_l, 
        c=cluster_local, 
        cmap="tab10",
        alpha=0.7,
        s=30,
        label=f"Clip {clip_idx} local"
    )

    ax2.scatter(
        x_l[frame_idx_local],
        y_l[frame_idx_local],
        c=[cluster_local[frame_idx_local]],
        cmap="tab10",
        edgecolor="black",
        s=100,
        alpha=1.0
    )
    local_title_str = f"Local PCA: Clip {clip_idx}, Frame {frame_idx_local}"
    if terminated:
        local_title_str += " (terminated)"
    ax2.set_title(local_title_str)
    ax2.set_xlabel(f"PC1 (local) ({var_lcoal[0]*100:.2f}%)")
    ax2.set_ylabel(f"PC2 (local) ({var_lcoal[1]*100:.2f}%)")

    plt.tight_layout()

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return img


def render_with_global_and_local_pca_progression(
    rollout: dict,
    walker_name: str,
    pca_global: np.ndarray,
    cluster_global: np.ndarray,
    global_subset_indices: np.ndarray,
    pca_local: np.ndarray,
    cluster_local: np.ndarray,
    var_lcoal: np.ndarray,
    var_global: np.ndarray,
):
    """
    Render the MuJoCo frames side-by-side with a figure showing
    BOTH global PCA + local PCA for each frame.
    """
    
    frames_mujoco = render_from_saved_rollout(rollout, walker_name)[1:]
    T = len(frames_mujoco)
    
    if T != len(pca_local):
        raise ValueError(f"Mismatch: {T} MuJoCo frames vs {len(pca_local)} local PCA steps.")
    if T != len(global_subset_indices):
        raise ValueError(f"Mismatch: {T} frames vs {len(global_subset_indices)} global_subset_indices.")

    clip_idx = int(rollout["info"][0]["clip_idx"])
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    worker_args = []
    for frame_idx in range(T):
        worker_args.append((
            frame_idx,
            pca_global,
            cluster_global,
            global_subset_indices,
            pca_local,
            cluster_local,
            T,
            clip_idx,
            var_lcoal,
            var_global,
        ))
    
    print("Rendering PCA (global+local) progression...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_pca = pool.starmap(global_local_pca_worker, worker_args)
        
    concat_frames = []
    for idx in range(T):
        concat_img = np.hstack([frames_mujoco[idx], frames_pca[idx]])
        concat_frames.append(concat_img)

    matplotlib.use(orig_backend)
    return concat_frames


def plot_homology(fly_activations, rodent_activations, layer_idx, sample_size=500):
    '''
    Plotting the homology of data, trying to find topological informations
    - H0 (Number of connected components, i.e clusters), this detects distinct activation states
    - H1 (Number of loops), this identifies cyclic behaviors (e.g., locomotion)
    - H2 (Number of voids), this finds higher-order structure (e.g., transitions between states, transient loops (from stopping to turning))
    '''
    scaler = StandardScaler()
    pca = PCA(n_components=10)
    fly_sampled = subsample_data(fly_activations[layer_idx], sample_size=sample_size)
    fly_scaled = scaler.fit_transform(fly_sampled)
    fly_reduced = pca.fit_transform(fly_scaled)

    rodent_sampled = subsample_data(rodent_activations[layer_idx], sample_size=sample_size)
    rodent_scaled = scaler.fit_transform(rodent_sampled)
    rodent_reduced = pca.fit_transform(rodent_scaled)
    print(f"Explained variance (Fly): {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Explained variance (Rodent): {np.sum(pca.explained_variance_ratio_):.4f}")

    fly_diagrams = ripser(fly_reduced, maxdim=2)['dgms']
    rodent_diagrams = ripser(rodent_reduced, maxdim=2)['dgms']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_diagrams(fly_diagrams, ax=axes[0])
    axes[0].set_title(f"Fly - Persistence Diagram for {layer_idx[0]} - {layer_idx[1]}")

    plot_diagrams(rodent_diagrams, ax=axes[1])
    axes[1].set_title(f"Rodent - Persistence Diagram for {layer_idx[0]} - {layer_idx[1]}")

    plt.show()