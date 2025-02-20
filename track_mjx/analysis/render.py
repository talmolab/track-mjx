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

from scipy.ndimage import gaussian_filter1d

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


def compute_forward_velocity(qposes, dt, smooth_sigma=2):
    """Computes and smooths forward velocity using central differencing."""
    
    com_positions = qposes[:, 0]
    raw_velocities = (com_positions[2:] - com_positions[:-2]) / (2 * dt)  # central difference
    raw_velocities = np.insert(raw_velocities, 0, raw_velocities[0]) 
    raw_velocities = np.append(raw_velocities, raw_velocities[-1])
    return gaussian_filter1d(raw_velocities, sigma=smooth_sigma)


def compute_height_based_leg_phases(qposes, leg_indices, threshold=0.05):
    """Determines swing (1) or stance (0) phases for each leg tip."""
    
    leg_heights = qposes[:, leg_indices]
    raw_leg_phases = (leg_heights > threshold).astype(int) # > is swing, <= is stance
    return (raw_leg_phases > 0.5).astype(int)  # re-binarize


def plot_height_based_gait_analysis(qposes, leg_indices, leg_labels, dt, clip_start, num_clips, timesteps_per_clip, color, title_prefix):
    """Plots multiple clips side-by-side using height based GRF model"""
    
    fig, axes = plt.subplots(2, num_clips, figsize=(5 * num_clips, 6), gridspec_kw={'height_ratios': [1, 3]})

    for i in range(num_clips):
        clip_id = clip_start + i
        start_idx = clip_id * timesteps_per_clip
        end_idx = start_idx + timesteps_per_clip
        qposes_clip = qposes[start_idx:end_idx, :]

        forward_velocity = compute_forward_velocity(qposes_clip, dt)
        leg_phases = compute_height_based_leg_phases(qposes_clip, leg_indices)
        
        time_axis = np.linspace(0, timesteps_per_clip * dt, timesteps_per_clip)

        axes[0, i].plot(time_axis, forward_velocity, color=color, linewidth=1)
        axes[0, i].set_ylabel("Velocity (mm/s)")
        axes[0, i].set_xticks([])
        axes[0, i].set_xlim(0, time_axis[-1])
        axes[0, i].set_title(f"{title_prefix} - Clip {clip_id}")

        im = axes[1, i].imshow(leg_phases.T, cmap="gray_r", aspect="auto", interpolation="nearest")
        axes[1, i].set_yticks(np.arange(len(leg_labels)))
        axes[1, i].set_yticklabels(leg_labels)
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_xticks(np.linspace(0, timesteps_per_clip - 1, 6))
        axes[1, i].set_xticklabels(np.round(np.linspace(0, timesteps_per_clip * dt, 6), 2))
        
        if i == num_clips - 1:
            legend_patches = [plt.Line2D([0], [0], color="black", lw=4, label="Swing"),
                              plt.Line2D([0], [0], color="white", lw=4, label="Stance")]
            axes[1, i].legend(handles=legend_patches, loc="upper right", frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.show()


def compute_joint_torques(model, data, qpos, qvel, qacc, valid_joint_indices):
    """
    Computes joint torques using MuJoCo's inverse dynamics, keeping only `-0` joints.
    """
    expected_dim = len(valid_joint_indices)
    actual_dim = len(qpos)

    if actual_dim != expected_dim:
        raise ValueError(f"Mismatch: Expected {expected_dim} qpos, got {actual_dim}.")

    data.qpos.fill(0)
    data.qvel.fill(0)
    data.qacc.fill(0)

    # assign only to the valid joints
    data.qpos[valid_joint_indices] = qpos
    data.qvel[valid_joint_indices] = qvel
    data.qacc[valid_joint_indices] = qacc

    mujoco.mj_inverse(model, data)
    return np.copy(data.qfrc_inverse[valid_joint_indices])  # extract only valid joint torques


def process_torque_estimation(model, qposes_rollout, dt, indices, valid_joint_indices):
    """
    Worker function to estimate torques for a subset of time steps.
    """
    data = mujoco.MjData(model)
    torques = np.zeros((len(indices), len(valid_joint_indices)))

    for i, t in enumerate(indices):
        qpos = qposes_rollout[t, valid_joint_indices]  # only valid indexes are used
        qvel = (qposes_rollout[t + 1, valid_joint_indices] - qposes_rollout[t - 1, valid_joint_indices]) / (2 * dt)
        qacc = (qposes_rollout[t + 1, valid_joint_indices] - 2 * qposes_rollout[t, valid_joint_indices] + qposes_rollout[t - 1, valid_joint_indices]) / (dt ** 2)

        torques[i] = compute_joint_torques(model, data, qpos, qvel, qacc, valid_joint_indices)

    return torques


def estimate_joint_forces_parallel(qposes_rollout, dt, walker_type, num_workers=mp.cpu_count()):
    """
    Estimates joint forces using parallel processing, using only `-0` joints.
    """
    num_steps = qposes_rollout.shape[0]
    step_indices = np.array_split(range(1, num_steps - 1), num_workers)
    
    # valid index comes from configs
    if walker_type=='fly':
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
    elif walker_type=='rodent':
        pair_render_xml_path = str(
        (
            Path(__file__).parent
            / ".."
            / "environment"
            / "walker"
            / "assets"
            / "rodent"
            / "rodent_ghostpair_spec.xml"
        ).resolve()
        )

    model = mujoco.MjModel.from_xml_path(pair_render_xml_path)

    # extract only joints ending with `-0`
    valid_joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name.endswith("-0")]
    joint_name_to_index = {model.joint(i).name: i for i in range(model.njnt) if model.joint(i).name.endswith("-0")}
    valid_joint_indices = np.array([joint_name_to_index[name] for name in valid_joint_names])
    print(f"Using {len(valid_joint_indices)} valid joints out of {model.njnt}")

    with mp.Pool(num_workers) as pool:
        results = pool.starmap(process_torque_estimation, [(model, qposes_rollout, dt, indices, valid_joint_indices) for indices in step_indices])

    return np.vstack(results)


def compute_leg_phases_from_joint_forces(joint_forces, contact_threshold=50):
    """
    Determines swing (1) or stance (0) phases for each leg based on joint forces.
    """
    if joint_forces.ndim == 1:  
        joint_forces = joint_forces[:, np.newaxis]  # Ensure shape (T, 1)

    # stance (force above threshold) = 0, swing (force below threshold) = 1
    leg_phases = (np.abs(joint_forces) < contact_threshold).astype(int)
    return leg_phases


def plot_force_based_gait_analysis(joint_forces, leg_labels, dt, clip_start, num_clips, timesteps_per_clip, color, title_prefix, contact_threshold):
    """
    Plots multiple GRF clips with force based measures
    """

    fig, axes = plt.subplots(2, num_clips, figsize=(5 * num_clips, 6), gridspec_kw={'height_ratios': [1, 3]})

    for i in tqdm(range(num_clips)):
        clip_id = clip_start + i
        start_idx = clip_id * timesteps_per_clip
        end_idx = min(start_idx + timesteps_per_clip, len(joint_forces))

        force_clip = joint_forces[start_idx:end_idx]

        if force_clip.ndim == 1:
            force_clip = force_clip[:, np.newaxis]  

        time_axis = np.linspace(0, len(force_clip) * dt, len(force_clip))

        # smoothed joint torques
        summed_forces = np.sum(force_clip, axis=1) if force_clip.shape[1] > 1 else force_clip.squeeze()
        smoothed_forces = gaussian_filter1d(summed_forces, sigma=2)

        axes[0, i].plot(time_axis, smoothed_forces, color=color, linewidth=1)
        axes[0, i].set_ylabel("Joint Torque (Nm)")
        axes[0, i].set_xticks([])
        axes[0, i].set_xlim(0, time_axis[-1])
        axes[0, i].set_title(f"{title_prefix} - Clip {clip_id}")

        # leg phases from estimated joint forces
        phase_clip = compute_leg_phases_from_joint_forces(force_clip, contact_threshold)

        im = axes[1, i].imshow(phase_clip.T, cmap="gray_r", aspect="auto", interpolation="nearest")
        axes[1, i].set_yticks(np.arange(len(leg_labels)))
        axes[1, i].set_yticklabels(leg_labels)
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_xticks(np.linspace(0, timesteps_per_clip - 1, 6))
        axes[1, i].set_xticklabels(np.round(np.linspace(0, timesteps_per_clip * dt, 6), 2))

        if i == num_clips - 1:
            legend_patches = [plt.Line2D([0], [0], color="black", lw=4, label="Swing"),
                              plt.Line2D([0], [0], color="white", lw=4, label="Stance")]
            axes[1, i].legend(handles=legend_patches, loc="upper right", frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.show()
    

# def estimate_ground_forces_per_joint(qposes_rollout, dt, mass_per_joint, z_indices):
#     """
#     Approximates ground reaction forces for each joint using second derivative of Z positions.
#     """
#     # z-axis positions for the specified joints
#     z_positions = qposes_rollout[:, z_indices]  # Shape (T, J_selected)

#     # velocities and accelerations along Z-axis
#     z_velocities = np.gradient(z_positions, dt, axis=0)
#     z_accelerations = np.gradient(z_velocities, dt, axis=0)

#     # GRF per joint
#     ground_forces = mass_per_joint * (z_accelerations + 9.81)  # F = m*a + gravity
#     return ground_forces


# def compute_leg_phases_from_grf_per_joint(ground_forces_per_joint, contact_threshold=1.0):
#     """
#     Determines swing (1) or stance (0) phases per joint based on ground reaction forces.
#     """
#     # stance (force above threshold) = 0, swing (force below threshold) = 1
#     leg_phases = (np.abs(ground_forces_per_joint) < contact_threshold).astype(int)
    
#     return leg_phases  # shape (T, J)


# def plot_grf_based_gait_analysis_per_joint(ground_forces_per_joint, leg_labels, dt, clip_start, num_clips, timesteps_per_clip, color, title_prefix, contact_threshold):
#     """
#     Plots multiple GRF clips side-by-side, showing GRF per joint using calculation of the force pushing downward on the ground on the z-components.
#     """

#     fig, axes = plt.subplots(2, num_clips, figsize=(5 * num_clips, 6), gridspec_kw={'height_ratios': [1, 3]})

#     for i in range(num_clips):
#         clip_id = clip_start + i
#         start_idx = clip_id * timesteps_per_clip
#         end_idx = min(start_idx + timesteps_per_clip, len(ground_forces_per_joint))

#         grf_clip = ground_forces_per_joint[start_idx:end_idx]

#         time_axis = np.linspace(0, len(grf_clip) * dt, len(grf_clip))

#         for j in range(grf_clip.shape[1]):  # loop over joints
#             axes[0, i].plot(time_axis, grf_clip[:, j], color=color, alpha=0.6, linewidth=1, label=f"Joint {leg_labels[j]}" if i == 0 else "")

#         axes[0, i].set_ylabel("GRF per Joint (N)")
#         axes[0, i].set_xticks([])
#         axes[0, i].set_xlim(0, time_axis[-1])
#         axes[0, i].set_title(f"{title_prefix} - Clip {clip_id}")

#         # stance/swing per joint
#         phase_clip = compute_leg_phases_from_grf_per_joint(grf_clip, contact_threshold)

#         # stance/swing
#         im = axes[1, i].imshow(phase_clip.T, cmap="gray_r", aspect="auto", interpolation="nearest")
#         axes[1, i].set_yticks(np.arange(len(leg_labels)))
#         axes[1, i].set_yticklabels(leg_labels)
#         axes[1, i].set_xlabel("Time (s)")
#         axes[1, i].set_xticks(np.linspace(0, timesteps_per_clip - 1, 6))
#         axes[1, i].set_xticklabels(np.round(np.linspace(0, timesteps_per_clip * dt, 6), 2))

#         if i == num_clips - 1:
#             legend_patches = [plt.Line2D([0], [0], color="black", lw=4, label="Swing"),
#                               plt.Line2D([0], [0], color="white", lw=4, label="Stance")]
#             axes[1, i].legend(handles=legend_patches, loc="upper right", frameon=True, edgecolor="black")

#     plt.tight_layout()
#     plt.show()


def estimate_ground_forces_per_joint(qposes_rollout, dt, mass_per_joint, z_indices, smooth_sigma=5):
    """
    Approximates ground reaction forces for each joint using second derivative of Z positions.
    Includes noise reduction techniques (thresholding binary masks & smoothing).
    """
   
    z_positions = qposes_rollout[:, z_indices]  # Shape (T, J_selected)

    # velocities and accelerations along Z-axis
    z_velocities = np.gradient(z_positions, dt, axis=0)
    z_accelerations = np.gradient(z_velocities, dt, axis=0)

    # raw GRF per joint
    ground_forces = mass_per_joint * (z_accelerations + 9.81)  # F = m*a + gravity

    # binary masks
    ground_forces[np.abs(ground_forces) < 1e-4] = 0

    # smooth GRF with Gaussian filter
    ground_forces_smooth = gaussian_filter1d(ground_forces, sigma=smooth_sigma, axis=0)

    return ground_forces_smooth


def compute_leg_phases_from_grf_per_joint(ground_forces_per_joint, contact_threshold=1.0, min_duration=10):
    """
    Determines swing (1) or stance (0) phases per joint based on ground reaction forces.
    Uses thresholding and short-event filtering.
    """
    # convert GRF into stance/swing binary
    leg_phases = (np.abs(ground_forces_per_joint) > contact_threshold).astype(int)

    # remove short stance/swing events (transient noise)
    for joint in range(leg_phases.shape[1]):
        for t in range(1, len(leg_phases) - 1):
            if leg_phases[t, joint] != leg_phases[t - 1, joint]:
                start = t
                while t < len(leg_phases) - 1 and leg_phases[t, joint] == leg_phases[start, joint]:
                    t += 1
                duration = t - start
                if duration < min_duration:
                    leg_phases[start:t, joint] = 1 - leg_phases[start, joint]  # Flip small transitions

    return leg_phases


def plot_grf_based_gait_analysis_per_joint(ground_forces_per_joint, leg_labels, dt, clip_start, num_clips, timesteps_per_clip, color, title_prefix, contact_threshold):
    """
    Plots multiple GRF clips side-by-side, showing GRF per joint using calculation of the force pushing downward on the ground.
    """

    fig, axes = plt.subplots(2, num_clips, figsize=(5 * num_clips, 6), gridspec_kw={'height_ratios': [1, 3]})

    for i in range(num_clips):
        clip_id = clip_start + i
        start_idx = clip_id * timesteps_per_clip
        end_idx = min(start_idx + timesteps_per_clip, len(ground_forces_per_joint))

        grf_clip = ground_forces_per_joint[start_idx:end_idx]
        phase_clip = compute_leg_phases_from_grf_per_joint(grf_clip, contact_threshold)

        time_axis = np.linspace(0, len(grf_clip) * dt, len(grf_clip))

        # GRF per joint
        for j in range(grf_clip.shape[1]):  
            axes[0, i].plot(time_axis, grf_clip[:, j], color=color, alpha=0.6, linewidth=1, label=f"Joint {leg_labels[j]}" if i == 0 else "")

        axes[0, i].set_ylabel("GRF per Joint (N)")
        axes[0, i].set_xticks([])
        axes[0, i].set_xlim(0, time_axis[-1])
        axes[0, i].set_title(f"{title_prefix} - Clip {clip_id}")

        # Stance/Swing per joint
        im = axes[1, i].imshow(phase_clip.T, cmap="gray_r", aspect="auto", interpolation="nearest")
        axes[1, i].set_yticks(np.arange(len(leg_labels)))
        axes[1, i].set_yticklabels(leg_labels)
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_xticks(np.linspace(0, timesteps_per_clip - 1, 6))
        axes[1, i].set_xticklabels(np.round(np.linspace(0, timesteps_per_clip * dt, 6), 2))

        if i == num_clips - 1:
            legend_patches = [plt.Line2D([0], [0], color="black", lw=4, label="Swing"),
                              plt.Line2D([0], [0], color="white", lw=4, label="Stance")]
            axes[1, i].legend(handles=legend_patches, loc="upper right", frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.show()



def estimate_intrinsic_dim(X, n_jobs=-1, sample_size=500):
    """intrinsic dimensionality using MLE and using parallelized LOF"""
    X_sample = subsample_data(X, sample_size)
    lof = LocalOutlierFactor(n_neighbors=10, metric='euclidean', n_jobs=n_jobs)
    return -lof.fit(X_sample).negative_outlier_factor_.mean()


def fast_isomap(X, n_components=2, n_neighbors=15, sample_size=500):
    """geodesic distances using Isomap on a subsample of the dataset for efficiency"""
    X_sample = subsample_data(X, sample_size)
    return Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X_sample)

    
def plot_homology(fly_activations, rodent_activations, layer_idx, sample_size=500):
    '''
    Plottin the homology of data, trying to find topological informations
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
    

def plot_intrinsic(fly_activations, rodent_activations, layer_idx, sample_size=500):
    '''
    Plotting the intrinsic dimensionality
    
    Check if fly and rodent activations lie on similar low-dimensional manifolds by
    estimating intrinsic dimensionality using MLE and visualize Isomap geodesics
    '''
    fly_dim = estimate_intrinsic_dim(fly_activations[layer_idx], sample_size=sample_size)
    rodent_dim = estimate_intrinsic_dim(rodent_activations[layer_idx], sample_size=sample_size)

    print(f"Estimated Intrinsic Dimensionality: Fly={fly_dim:.2f}, Rodent={rodent_dim:.2f}")

    iso_fly = fast_isomap(fly_activations[layer_idx], sample_size=sample_size)
    iso_rodent = fast_isomap(rodent_activations[layer_idx], sample_size=sample_size)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(iso_fly[:, 0], iso_fly[:, 1], alpha=0.5)
    axes[0].set_title(f"Fly - Isomap Projection for {layer_idx[0]} - {layer_idx[1]}")

    axes[1].scatter(iso_rodent[:, 0], iso_rodent[:, 1], alpha=0.5, color='orange')
    axes[1].set_title(f"Rodent - Isomap Projection for {layer_idx[0]} - {layer_idx[1]}")

    plt.show()