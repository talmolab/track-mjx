import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.colors import hsv_to_rgb
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

import ripser
from persim import plot_diagrams, wasserstein
import gudhi as gd
from gudhi.representations import Landscape, Silhouette, PersistenceImage
import umap
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def plot_peak_force_gait(peak_force_ctrl, leg_labels, dt, timesteps_per_clip, color, title_prefix, contact_threshold=None, already_binarized=False):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [1, 3]})

    time_axis = np.linspace(0, timesteps_per_clip * dt, timesteps_per_clip)

    if already_binarized:
        phase_clip = peak_force_ctrl.astype(int)
        total_force = np.sum(phase_clip, axis=1)
    else:
        smoothed_forces = peak_force_ctrl
        total_force = np.sum(smoothed_forces, axis=1)
        phase_clip = (np.abs(smoothed_forces) > contact_threshold).astype(int)

    axes[0].plot(time_axis, total_force, color=color, linewidth=1)
    axes[0].set_ylabel("Total Force (N)")
    axes[0].set_xticks([])
    axes[0].set_xlim(0, time_axis[-1])
    axes[0].set_title(f"{title_prefix}")

    im = axes[1].imshow(phase_clip.T, cmap="gray_r", aspect="auto", interpolation="nearest")
    axes[1].set_yticks(np.arange(len(leg_labels)))
    axes[1].set_yticklabels(leg_labels)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xticks(np.linspace(0, timesteps_per_clip - 1, 6))
    axes[1].set_xticklabels(np.round(np.linspace(0, timesteps_per_clip * dt, 6), 2))

    legend_patches = [
        plt.Line2D([0], [0], color="black", lw=4, label="Swing"),
        plt.Line2D([0], [0], color="white", lw=4, label="Stance")
    ]
    axes[1].legend(handles=legend_patches, loc="upper right", frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.show()


def create_time_windows(intentions, window_size=31, stride=1):
    """
    Create time windows from intention space data
    
    Parameters:
    - intentions: Array of shape (n_timesteps, n_features)
    - window_size: Number of frames in each window
    - stride: Step size between consecutive windows
    
    Returns:
    - Array of shape (n_windows, window_size * n_features)
    """
    n_timesteps, n_features = intentions.shape
    n_windows = (n_timesteps - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size * n_features))
    center_frames = np.zeros(n_windows, dtype=int)
    
    for i in range(n_windows):
        start_idx = i * stride
        # Extract window and flatten
        window = intentions[start_idx:start_idx+window_size]
        windows[i] = window.reshape(-1)
        # Calculate center frame
        center_frames[i] = start_idx + window_size//2
    
    return windows, center_frames


def sample_diverse_windows(windows, n_samples=10000, n_clusters=10):
    """
    Sample diverse windows using PCA and K-means clustering as in the paper
    
    Parameters:
    - windows: Array of windowed data
    - n_samples: Number of samples to extract
    - n_clusters: Number of clusters for K-means
    
    Returns:
    - Sampled windows and their indices
    """
    print(f"Sampling {n_samples} diverse windows from {len(windows)} total windows...")
    
    # have fewer windows than requested samples, return all windows
    if len(windows) <= n_samples:
        print(f"Only {len(windows)} windows available, using all of them")
        return windows, np.arange(len(windows))
    
    # PCA to reduce dimensionality (as in paper)
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=min(50, windows.shape[1]))  # Paper used 50 components
    reduced_windows = pca.fit_transform(windows)
    
    # cluster the reduced windows
    print(f"Clustering data into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(reduced_windows)
    
    # sample from each cluster proportionally
    sampled_indices = []
    
    for cluster_idx in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_idx)[0]
        
        # samples for this cluster (proportional to cluster size)
        n_cluster_samples = int(n_samples * len(cluster_indices) / len(windows))
        n_cluster_samples = max(1, n_cluster_samples)  # At least 1 sample per cluster
        
        # sample indices from this cluster
        if n_cluster_samples >= len(cluster_indices):
            # iIf we need more samples than available, sample with replacement
            cluster_sampled = np.random.choice(
                cluster_indices, 
                size=n_cluster_samples, 
                replace=True
            )
        else:
            # sample without replacement
            cluster_sampled = np.random.choice(
                cluster_indices, 
                size=n_cluster_samples, 
                replace=False
            )
        
        sampled_indices.extend(cluster_sampled)
    
    if len(sampled_indices) > n_samples:
        # too many samples, randomly subsample
        sampled_indices = np.random.choice(sampled_indices, size=n_samples, replace=False)
    elif len(sampled_indices) < n_samples:
        # too few samples, add more from random clusters
        remaining = n_samples - len(sampled_indices)
        remaining_indices = list(set(range(len(windows))) - set(sampled_indices))
        
        if len(remaining_indices) >= remaining:
            # have enough, sample without replacement
            additional = np.random.choice(remaining_indices, size=remaining, replace=False)
        else:
            # sample with replacement from all indices
            additional = np.random.choice(range(len(windows)), size=remaining, replace=True)
            
        sampled_indices.extend(additional)
    
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)
    
    return windows[sampled_indices], sampled_indices


def apply_umap_to_intention_windows(windows, n_components=3):
    """
    Apply UMAP to intention space windows with robust error handling
    
    Parameters:
    - windows: Array of time windows from intention space
    - n_components: Number of UMAP dimensions
    
    Returns:
    - UMAP embedding
    """
    
    if len(windows) < 10:
        print("Too few windows for UMAP, falling back to PCA")
        pca = PCA(n_components=n_components)
        return pca.fit_transform(StandardScaler().fit_transform(windows))
    
    scaler = StandardScaler()
    windows_scaled = scaler.fit_transform(windows)
    
    n_neighbors = min(20, max(2, len(windows) // 5))
    
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    
    embedded = umap_reducer.fit_transform(windows_scaled)
    return embedded
    

def create_color_by_phase(phases):
    """
    Convert phases to colors using HSV colormap with robust error handling
    
    Parameters:
    - phases: Array of phase values
    
    Returns:
    - Array of RGB colors
    """
    # Normalize phases to range [0, 1] for hue
    # First ensure phases are in proper range
    phases = np.asarray(phases)
    phases = np.mod(phases, 2*np.pi)  # Ensure in range [0, 2π]
    hues = phases / (2 * np.pi)  # Convert to range [0, 1]
    
    hsv_colors = np.zeros((len(hues), 3))
    hsv_colors[:, 0] = hues  # Hue
    hsv_colors[:, 1] = 0.8   # Saturation
    hsv_colors[:, 2] = 0.8   # Value
    
    try:
        rgb_colors = hsv_to_rgb(hsv_colors)
        # Ensure all values are within 0-1 range
        rgb_colors = np.clip(rgb_colors, 0, 1)
        return rgb_colors
    except:
        # Fall back to a simple colormap
        print("Error converting HSV to RGB, using fallback colormap")
        return plt.cm.hsv(hues)


def visualize_manifold(embedded, gait_metrics, trajectory_indices=None):
    """
    Create visualizations matching eLife paper Figure 4 with robust color handling
    
    Parameters:
    - embedded: UMAP embedding
    - gait_metrics: Dictionary of extracted gait metrics
    - trajectory_indices: Indices for trajectory visualization (optional)
    """
    phases = gait_metrics['phases']
    frequencies = gait_metrics['frequencies']
    feet_in_stance = gait_metrics['feet_in_stance']
    
    if len(phases) != len(embedded):
        print(f"Warning: Number of gait metrics ({len(phases)}) doesn't match embedding points ({len(embedded)})")
        print("Adjusting data to match sizes...")
        
        min_size = min(len(embedded), len(phases))
        embedded = embedded[:min_size]
        phases = phases[:min_size]
        frequencies = frequencies[:min_size]
        feet_in_stance = feet_in_stance[:min_size]
    
    # figure with 2x2 layout (like Figure 4 in paper)
    fig = plt.figure(figsize=(20, 16))
    
    # calculate density for better visualization
    try:
        density = gaussian_kde(embedded.T)(embedded.T)
    except:
        print("Error calculating density, using uniform density")
        density = np.ones(len(embedded))
    
    idx = density.argsort()
    x, y, z = embedded[idx, 0], embedded[idx, 1], embedded[idx, 2]
    phases_sorted = phases[idx]
    
    frequencies = np.array(frequencies, dtype=float)
    frequencies = np.nan_to_num(frequencies, nan=1.0)  # Replace NaNs with 1.0
    frequencies = np.clip(frequencies, 0.1, 10.0)  # Reasonable range for frequencies
    frequencies_sorted = frequencies[idx]
    
    feet_in_stance = np.round(feet_in_stance).astype(int)
    feet_in_stance = np.clip(feet_in_stance, 0, 6)  # Limit to 0-6 range for feet
    feet_sorted = feet_in_stance[idx]
    
    # Plot 1: colored by frequency
    ax1 = fig.add_subplot(221, projection='3d')
    scatter1 = ax1.scatter(
        x, y, z,
        c=frequencies_sorted,
        cmap='viridis',
        alpha=0.6,
        s=5,
        vmin=min(0.5, frequencies.min()),
        vmax=max(5.0, frequencies.max())
    )
    ax1.set_title("Manifold Colored by Mean Frequency of Walking", fontsize=14)
    fig.colorbar(scatter1, ax=ax1, label="Frequency (Hz)")
    
    # Plot 2: colored by phase
    ax2 = fig.add_subplot(222, projection='3d')
    try:
        phase_colors = create_color_by_phase(phases_sorted)
        ax2.scatter(
            x, y, z,
            c=phase_colors,
            alpha=0.6,
            s=5
        )
    except:
        print("Error with phase colors, using viridis colormap instead")
        ax2.scatter(
            x, y, z,
            c=phases_sorted,
            cmap='viridis',
            alpha=0.6,
            s=5,
            vmin=-np.pi,
            vmax=np.pi
        )
    
    ax2.set_title("Manifold Colored by Global Phase", fontsize=14)
    
    # Plot 3: trajectories
    ax3 = fig.add_subplot(223, projection='3d')
    
    ax3.scatter(
        embedded[:, 0], 
        embedded[:, 1], 
        embedded[:, 2],
        color='lightgray',
        alpha=0.1,
        s=2
    )
    
    # if trajectory indices are provided, show those
    if trajectory_indices is not None and len(trajectory_indices) > 0:
        # ensure trajectory indices are valid
        valid_traj_indices = [i for i in trajectory_indices if i < len(embedded)]
        if len(valid_traj_indices) > 0:
            # plot single trajectory with gradient color
            trajectory = embedded[valid_traj_indices]
            for i in range(len(trajectory)-1):
                color = plt.cm.viridis(i/(len(trajectory)-1))
                ax3.plot(
                    trajectory[i:i+2, 0],
                    trajectory[i:i+2, 1],
                    trajectory[i:i+2, 2],
                    color=color,
                    linewidth=2
                )
            
            # mark start point
            ax3.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                color='black',
                s=50,
                marker='o',
                label='Start'
            )
    else:
        # find a few example trajectories & cluster the embedding to find different regions
        try:
            n_clusters = min(3, len(embedded) // 10)  # Ensure we don't try to make too many clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embedded)
            
            # find trajectory starting points near cluster centers
            colors = ['green', 'orange', 'purple']
            for i in range(n_clusters):
                # find points in this cluster
                cluster_points = np.where(clusters == i)[0]
                if len(cluster_points) < 30:
                    continue
                    
                # find point closest to cluster center
                center = kmeans.cluster_centers_[i]
                distances = np.sum((embedded[cluster_points] - center)**2, axis=1)
                closest_idx = cluster_points[np.argmin(distances)]
                
                # extract trajectory (30 points)
                traj_start = max(0, closest_idx - 15)
                traj_end = min(len(embedded), traj_start + 30)
                
                if traj_end - traj_start < 10:  # Skip if too short
                    continue
                    
                traj = np.arange(traj_start, traj_end)
                
                ax3.plot(
                    embedded[traj, 0],
                    embedded[traj, 1],
                    embedded[traj, 2],
                    color=colors[i % len(colors)],
                    linewidth=2
                )
                
                # ax3.scatter(
                #     embedded[traj[0], 0],
                #     embedded[traj[0], 1],
                #     embedded[traj[0], 2],
                #     color='black',
                #     s=40,
                #     marker='o'
                # )
                
        except Exception as e:
            print(f"Error finding trajectories: {e}")
    
    ax3.set_title("Example Trajectories", fontsize=14)
    
    # Plot 4: feet in stance
    ax4 = fig.add_subplot(224, projection='3d')
    scatter4 = ax4.scatter(
        x, y, z,
        c=feet_sorted,
        cmap='viridis',
        alpha=0.6,
        s=5,
        vmin=0,
        vmax=6
    )
    ax4.set_title("Manifold Colored by Number of Feet in Stance", fontsize=14)
    fig.colorbar(scatter4, ax=ax4, label="Feet in stance")
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.view_init(elev=20, azim=-60) 
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.grid(False)
    
    plt.tight_layout()
    plt.show()


def visualize_additional_gait_metrics(embedded, gait_metrics):
    """
    Visualize additional gait-related metrics on the manifold with robust error handling
    
    Parameters:
    - embedded: UMAP embedding
    - gait_metrics: Dictionary of gait metrics
    """

    if len(embedded) != len(gait_metrics['duty_factor']):
        print(f"Warning: Size mismatch - embedding: {len(embedded)}, gait metrics: {len(gait_metrics['duty_factor'])}")

        min_size = min(len(embedded), len(gait_metrics['duty_factor']))
        embedded = embedded[:min_size]
        duty_factor = gait_metrics['duty_factor'][:min_size]
        clip_indices = gait_metrics['clip_indices'][:min_size]
        asymmetry = gait_metrics['asymmetry'][:min_size]
    else:
        duty_factor = gait_metrics['duty_factor']
        clip_indices = gait_metrics['clip_indices']
        asymmetry = gait_metrics['asymmetry']
    
    duty_factor = np.clip(np.nan_to_num(duty_factor, nan=0.5), 0, 1)
    clip_indices = np.nan_to_num(clip_indices, nan=0).astype(int)
    asymmetry = np.clip(np.nan_to_num(asymmetry, nan=0), -1, 1)
    
    fig = plt.figure(figsize=(20, 12))
    
    try:
        density = gaussian_kde(embedded.T)(embedded.T)
    except:
        print("Error calculating density, using uniform density")
        density = np.ones(len(embedded))
    
    idx = density.argsort()
    x, y, z = embedded[idx, 0], embedded[idx, 1], embedded[idx, 2]
    duty_factor_sorted = duty_factor[idx]
    asymmetry_sorted = asymmetry[idx]
    
    # Plot 1: Duty factor
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(
        x, y, z,
        c=duty_factor_sorted,
        cmap='viridis',
        alpha=0.6,
        s=5,
        vmin=0,
        vmax=1
    )
    ax1.set_title("Manifold Colored by Duty Factor", fontsize=14)
    fig.colorbar(scatter1, ax=ax1, label="Duty factor")
    
    # Plot 2: Gait asymmetry
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(
        x, y, z,
        c=asymmetry_sorted,
        cmap='coolwarm',
        vmin=-0.3, vmax=0.3,
        alpha=0.6,
        s=5
    )
    ax2.set_title("Manifold Colored by Gait Asymmetry", fontsize=14)
    fig.colorbar(scatter2, ax=ax2, label="Asymmetry (L-R)")
    
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.grid(False)
    
    plt.tight_layout()
    plt.show()


def analyze_cyclic_structure(embedded, gait_metrics):
    """
    Analyze the cyclic structure of the manifold
    
    Parameters:
    - embedded: UMAP embedding
    - gait_metrics: Dictionary of gait metrics
    """
    min_size = min(len(embedded), len(gait_metrics['phases']))
    phases = gait_metrics['phases'][:min_size]
    frequencies = gait_metrics['frequencies'][:min_size]
    feet_in_stance = gait_metrics['feet_in_stance'][:min_size]
    duty_factor = gait_metrics['duty_factor'][:min_size]
    
    plt.figure(figsize=(16, 6))
    
    # Plot 1: global phase vs. feet in stance (similar to Fig 2F in paper)
    plt.subplot(131)
    phase_bins = np.linspace(-np.pi, np.pi, 20)
    binned_phases = np.digitize(phases, phase_bins) - 1
    
    # calculate statistics for each bin
    mean_feet = np.zeros(len(phase_bins)-1)
    std_feet = np.zeros(len(phase_bins)-1)
    
    for i in range(len(phase_bins)-1):
        bin_mask = binned_phases == i
        if np.sum(bin_mask) > 0:
            mean_feet[i] = np.mean(feet_in_stance[bin_mask])
            std_feet[i] = np.std(feet_in_stance[bin_mask])
    
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    plt.plot(bin_centers, mean_feet, 'b-', linewidth=2)
    plt.fill_between(
        bin_centers, 
        mean_feet - std_feet, 
        mean_feet + std_feet, 
        color='b', 
        alpha=0.2
    )
    
    plt.xlabel('Global Phase (radians)')
    plt.ylabel('Feet in Stance')
    plt.title('Feet in Stance vs. Phase')
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    # Plot 2: frequency distribution
    plt.subplot(132)
    plt.hist(frequencies, bins=30, density=True, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Walking Frequencies')
    
    
    z = np.polyfit(frequencies, duty_factor, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(np.min(frequencies), np.max(frequencies), 100)
    plt.plot(x_trend, p(x_trend), "r--", linewidth=2)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Duty Factor')
    plt.title('Duty Factor vs. Frequency')
    
    plt.tight_layout()
    plt.show()


def create_multi_angle_view(embedded, gait_metrics, single_trajectory=None):
    """
    Create multiple viewpoints of the manifold with proper color bars
    
    Parameters:
    - embedded: UMAP embedding
    - gait_metrics: Dictionary of gait metrics
    - single_trajectory: Optional indices for a single trajectory
    """
    min_size = min(len(embedded), len(gait_metrics['phases']))
    embedded = embedded[:min_size]
    phases = gait_metrics['phases'][:min_size]
    
    fig = plt.figure(figsize=(20, 14))  # Increased width for colorbar
    
    fig.suptitle('Intention Space Manifold from Multiple Viewpoints', 
                 fontsize=16, y=0.98)
    
    cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    
    phase_range = np.linspace(-np.pi, np.pi, 256)
    phase_colors = create_color_by_phase(phase_range)
    phase_colors_2d = phase_colors.reshape(1, -1, 3)
    cbar_ax.imshow(phase_colors_2d, aspect='auto', extent=[-np.pi, np.pi, 0, 1])
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar_ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    cbar_ax.set_title('Global Phase', fontsize=12)
    
    if single_trajectory is not None:
        legend_ax = fig.add_axes([0.93, 0.15, 0.03, 0.1])  # [left, bottom, width, height]
        legend_ax.axis('off')
        
        legend_lines = []
        legend_lines.append(plt.Line2D([0], [0], color='green', lw=2, label='Early in Trajectory'))
        legend_lines.append(plt.Line2D([0], [0], color='yellow', lw=2, label='Middle of Trajectory'))
        legend_lines.append(plt.Line2D([0], [0], color='purple', lw=2, label='Late in Trajectory'))
        legend_lines.append(plt.Line2D([0], [0], marker='o', color='black', label='Start Point', 
                                      linestyle='None', markersize=8))
        
        legend_ax.legend(handles=legend_lines, loc='center', frameon=True)
    
    explanation_text = """
    This visualization shows the intention space manifold colored by global phase.
    Points with the same color correspond to the same phase in the gait cycle.
    The trajectory shows a path through the manifold during continuous walking.
    Different viewing angles help visualize the 3D structure of the manifold.
    """
    fig.text(0.5, 0.02, explanation_text, horizontalalignment='center', fontsize=10)
    
    angles = [
        (20, -60),  # Similar to paper's view
        (0, 0),     # Front view
        (90, 0),    # Top view
        (0, 90)     # Side view
    ]
    
    for i, (elev, azim) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        scatter = ax.scatter(
            embedded[:, 0], 
            embedded[:, 1], 
            embedded[:, 2],
            c=create_color_by_phase(phases),
            alpha=0.3,
            s=3
        )
        
        # if single_trajectory is not None:
        #     valid_traj = [idx for idx in single_trajectory if idx < len(embedded)]
            
        #     if len(valid_traj) > 1:
        #         trajectory = embedded[valid_traj]
                
               
        #         for j in range(len(trajectory)-1):
        #             color = plt.cm.viridis(j/(len(trajectory)-1))
        #             ax.plot(
        #                 trajectory[j:j+2, 0],
        #                 trajectory[j:j+2, 1],
        #                 trajectory[j:j+2, 2],
        #                 color=color,
        #                 linewidth=2
        #             )
                
        #         ax.scatter(
        #             trajectory[0, 0], 
        #             trajectory[0, 1], 
        #             trajectory[0, 2],
        #             color='black', 
        #             s=50, 
        #             marker='o'
        #         )
                
        #         # add arrow to show direction
        #         if len(trajectory) > 10:
        #             mid_point = len(trajectory) // 2
        #             ax.quiver(
        #                 trajectory[mid_point, 0],
        #                 trajectory[mid_point, 1],
        #                 trajectory[mid_point, 2],
        #                 trajectory[mid_point+1, 0] - trajectory[mid_point, 0],
        #                 trajectory[mid_point+1, 1] - trajectory[mid_point, 1],
        #                 trajectory[mid_point+1, 2] - trajectory[mid_point, 2],
        #                 color='red',
        #                 arrow_length_ratio=0.3,
        #                 linewidth=3
        #             )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'View: elev={elev}°, azim={azim}°')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.grid(False)
    
    plt.tight_layout(rect=[0, 0.05, 0.92, 0.95])  # Adjust layout to make room for colorbar
    plt.show()


def identify_trajectory_indices(embedded, start_idx=None, length=100):
    """
    Identify indices for a coherent trajectory through the manifold
    
    Parameters:
    - embedded: UMAP embedding
    - start_idx: Optional starting index (if None, one will be chosen)
    - length: Desired trajectory length
    
    Returns:
    - Array of trajectory indices
    """
    n_points = embedded.shape[0]
    
    # if no start index specified, choose one near a dense region
    if start_idx is None:
        # compute density and choose a point in a relatively dense area
        density = gaussian_kde(embedded.T)(embedded.T)
        # don't pick the densest point (might be in a crowded area), instead pick one around the 75th percentile of density
        density_threshold = np.percentile(density, 75)
        candidates = np.where(density > density_threshold)[0]
        if len(candidates) > 0:
            start_idx = np.random.choice(candidates)
        else:
            start_idx = np.random.randint(0, n_points)
    
    start_idx = max(0, min(start_idx, n_points - 1))
    
    if start_idx + length > n_points:
        length = n_points - start_idx
    
    trajectory_indices = np.arange(start_idx, start_idx + length)
    
    return trajectory_indices


def analyze_gait_cluster_distribution(embedded, gait_metrics, n_clusters=4):
    """
    Analyze how different gait patterns are distributed on the manifold
    
    Parameters:
    - embedded: UMAP embedding
    - gait_metrics: Dictionary of gait metrics
    - n_clusters: Number of potential gait clusters to find
    
    Returns:
    - Cluster labels
    """
    min_size = min(len(embedded), len(gait_metrics['phases']))
    embedded = embedded[:min_size]
    
    phases = gait_metrics['phases'][:min_size]
    frequencies = gait_metrics['frequencies'][:min_size]
    feet_in_stance = gait_metrics['feet_in_stance'][:min_size]
    duty_factor = gait_metrics['duty_factor'][:min_size]
    
    features = np.column_stack([
        frequencies,
        duty_factor,
        feet_in_stance
    ])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    n_clusters = min(n_clusters, len(embedded) // 10)
    n_clusters = max(2, n_clusters)  # At least 2 clusters
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: clusters on manifold
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(
        embedded[:, 0], 
        embedded[:, 1], 
        embedded[:, 2],
        c=cluster_labels,
        cmap='tab10',
        alpha=0.6,
        s=5
    )
    ax1.set_title("Gait Clusters on Manifold", fontsize=14)
    
    handles = []
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        percentage = 100 * count / len(cluster_labels)
        handles.append(plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            markerfacecolor=plt.cm.tab10(i/10), 
            markersize=10,
            label=f'Cluster {i+1}: {percentage:.1f}%'
        ))
    ax1.legend(handles=handles, loc='upper right')
    
    # Plot 2: frequency distribution by cluster
    ax2 = fig.add_subplot(232)
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 0:
            ax2.hist(
                frequencies[mask], 
                bins=15, 
                alpha=0.5, 
                label=f'Cluster {i+1}',
                color=plt.cm.tab10(i/10)
            )
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Count')
    ax2.set_title('Frequency Distribution by Cluster')
    ax2.legend()
    
    # Plot 3: duty factor by cluster
    ax3 = fig.add_subplot(233)
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 0:
            ax3.hist(
                duty_factor[mask], 
                bins=15, 
                alpha=0.5, 
                label=f'Cluster {i+1}',
                color=plt.cm.tab10(i/10)
            )
    ax3.set_xlabel('Duty Factor')
    ax3.set_ylabel('Count')
    ax3.set_title('Duty Factor Distribution by Cluster')
    ax3.legend()
    
    # Plot 4: feet in stance by cluster
    ax4 = fig.add_subplot(234)
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 0:
            ax4.hist(
                feet_in_stance[mask], 
                bins=15, 
                alpha=0.5, 
                label=f'Cluster {i+1}',
                color=plt.cm.tab10(i/10)
            )
    ax4.set_xlabel('Feet in Stance')
    ax4.set_ylabel('Count')
    ax4.set_title('Feet in Stance Distribution by Cluster')
    ax4.legend()
    
    # Plot 5: scatter plot of frequency vs. duty factor colored by cluster
    ax5 = fig.add_subplot(235)
    scatter5 = ax5.scatter(
        frequencies, 
        duty_factor,
        c=cluster_labels,
        cmap='tab10',
        alpha=0.6,
        s=5
    )
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Duty Factor')
    ax5.set_title('Frequency vs. Duty Factor')
    
    # Plot 6: phase histogram by cluster (if relevant)
    ax6 = fig.add_subplot(236, projection='polar')
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 0:
            hist, bins = np.histogram(
                phases[mask], 
                bins=16, 
                range=(-np.pi, np.pi)
            )
            width = (bins[1] - bins[0])
            ax6.bar(
                bins[:-1], 
                hist/np.sum(hist), 
                width=width, 
                alpha=0.5,
                label=f'Cluster {i+1}',
                color=plt.cm.tab10(i/10)
            )
    ax6.set_title('Phase Distribution by Cluster')
    ax6.legend(loc='upper right')
    
    ax1.view_init(elev=20, azim=-60)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_zlabel("UMAP 3")
    ax1.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return cluster_labels


def analyze_intention_gait_relationship(results, window_size):
    """
    Analyze the relationship between intention signals and gait parameters
    
    Parameters:
    - results: Results dictionary from main function
    """
    embedded = results['embedded']
    windows = results['windows']
    gait_metrics = results['gait_metrics']
    cluster_labels = results['cluster_labels']
    
    if windows.shape[0] != embedded.shape[0]:
        print(f"Warning: Size mismatch - windows: {windows.shape[0]}, embedding: {embedded.shape[0]}")
        min_size = min(windows.shape[0], embedded.shape[0])
        windows = windows[:min_size]
        embedded = embedded[:min_size]
        if len(cluster_labels) > min_size:
            cluster_labels = cluster_labels[:min_size]
    
    n_features = windows.shape[1] // window_size
    
    reshaped_windows = windows.reshape(windows.shape[0], window_size, n_features)
    
    center_idx = window_size // 2
    center_intentions = reshaped_windows[:, center_idx, :]
    
    gait_params = ['frequencies', 'feet_in_stance', 'duty_factor']
    correlations = {}
    
    for param in gait_params:
        if param in gait_metrics and len(gait_metrics[param]) == len(center_intentions):
            corr_matrix = np.zeros(n_features)
            p_values = np.zeros(n_features)
            
            for i in range(n_features):
                try:
                    corr_matrix[i], p_values[i] = pearsonr(
                        center_intentions[:, i], 
                        gait_metrics[param]
                    )
                except:
                    corr_matrix[i] = 0
                    p_values[i] = 1
            
            correlations[param] = (corr_matrix, p_values)
    
    if correlations:
        plt.figure(figsize=(15, 10))
        
        # colormap that marks statistical significance
        significant_threshold = 0.01
        
        for i, param in enumerate(correlations.keys()):
            corr_matrix, p_values = correlations[param]
            
            plt.subplot(len(correlations), 1, i+1)
            
            colors = ['lightgray' if p > significant_threshold else 'green' if c > 0 else 'red' 
                    for c, p in zip(corr_matrix, p_values)]
            
            plt.bar(range(n_features), corr_matrix, color=colors)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title(f'Correlation: Intention Signals vs {param}')
            plt.xlabel('Intention Signal Index')
            plt.ylabel('Correlation Coefficient')
            plt.ylim(-1, 1)

            for j, (corr, pval) in enumerate(zip(corr_matrix, p_values)):
                if pval < significant_threshold:
                    plt.text(j, corr * 0.9, f"{corr:.2f}*", 
                            ha='center', va='center', 
                            fontsize=8, color='black',
                            fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        

def compute_phase_using_hilbert(intentions, fps=30):
    """
    Compute phases using the discrete-time analytic signal method from the paper
    
    Parameters:
    - intentions: Array of shape (n_timesteps, n_features)
    - window_size: Size of time windows
    - fps: Frames per second
    
    Returns:
    - Dictionary with phases and frequencies for each component
    """
    n_timesteps, n_features = intentions.shape
    
    all_phases = np.zeros((n_timesteps, n_features))
    all_frequencies = np.zeros((n_timesteps, n_features))
    
    # process each component separately
    for i in range(n_features):
        # extract this component's time series
        signal = intentions[:, i]
        
        # filter the signal as in the paper (bandpass filter)
        b, a = butter(4, [0.5, 10], fs=fps, btype='band')
        try:
            filtered_signal = filtfilt(b, a, signal)
        except ValueError:
            # for short signals, use a simpler filter
            filtered_signal = signal - np.mean(signal)
            
        # Hilbert transform to compute the analytic signal
        try:
            analytic_signal = hilbert(filtered_signal)
            
            # extract instantaneous phase and unwrapped phase
            instantaneous_phase = np.angle(analytic_signal)
            unwrapped_phase = np.unwrap(instantaneous_phase)
            
            # instantaneous frequency from unwrapped phase
            instantaneous_frequency = np.diff(unwrapped_phase) / (2.0 * np.pi) * fps
            instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])
            
            # smooth frequency with Savitzky-Golay filter as in the paper
            window_length = min(15, len(instantaneous_frequency) // 2)
            if window_length > 3 and window_length % 2 == 1:
                from scipy.signal import savgol_filter
                instantaneous_frequency = savgol_filter(instantaneous_frequency, window_length, 3)
            
            all_phases[:, i] = instantaneous_phase
            all_frequencies[:, i] = instantaneous_frequency
            
        except:
            all_phases[:, i] = np.linspace(-np.pi, np.pi, n_timesteps)
            all_frequencies[:, i] = np.ones(n_timesteps)
    
    return {
        'phases': all_phases,
        'frequencies': all_frequencies
    }


def estimate_gait_metrics(cfrc, center_frames, window_size=31, fps=30):
    """
    Extract gait metrics from force data using methods from the paper
    
    Parameters:
    - cfrc: Array of shape (n_clips, n_frames, n_bodies, 6) containing force data
    - center_frames: Center frame indices for each window
    - window_size: Size of time windows
    - fps: Frames per second
    
    Returns:
    - Dictionary with gait metrics
    """
    frame_mapping = {}
    global_idx = 0
    
    for clip_idx in range(cfrc.shape[0]):
        n_frames = cfrc[clip_idx].shape[0]
        for frame_idx in range(n_frames):
            frame_mapping[global_idx] = (clip_idx, frame_idx)
            global_idx += 1
    
    phases = []
    frequencies = []
    feet_in_stance = []
    stance_durations = []
    swing_durations = []
    duty_factors = []
    asymmetry_values = []
    clip_indices = []
    valid_center_frames = []
    
    for frame_idx, center_frame in enumerate(center_frames):
        if center_frame in frame_mapping:
            clip_idx, clip_frame = frame_mapping[center_frame]
            
            clip_data = cfrc[clip_idx]
            feet_forces = clip_data
            
            half_window = window_size // 2
            if clip_frame >= half_window and clip_frame < clip_data.shape[0] - half_window:
                try:
                    # threshold force indicating stance
                    contact_threshold = 0.1
                    contacts = feet_forces > contact_threshold
                    
                    # get number of feet in stance at center frame
                    feet_in_stance.append(np.sum(contacts[clip_frame]))
                    
                    # calculate duty factor (as in the paper)
                    # extract a longer window for better stance/swing detection
                    extended_start = max(0, clip_frame - window_size)
                    extended_end = min(clip_data.shape[0], clip_frame + window_size)
                    extended_contacts = contacts[extended_start:extended_end]
                    
                    # calculate stance and swing durations
                    stance_duration = []
                    swing_duration = []
                    
                    for foot in range(extended_contacts.shape[1]):
                        foot_contacts = extended_contacts[:, foot]
                        
                        # find transitions (0->1 is swing->stance, 1->0 is stance->swing)
                        transitions = np.diff(foot_contacts.astype(int))
                        swing_to_stance = np.where(transitions == 1)[0]
                        stance_to_swing = np.where(transitions == -1)[0]
                        
                        if len(swing_to_stance) > 0 and len(stance_to_swing) > 0:
                            # calculate durations of complete stance and swing phases
                            complete_stances = []
                            complete_swings = []
                            
                            for i in range(len(swing_to_stance)):
                                for j in range(len(stance_to_swing)):
                                    if stance_to_swing[j] > swing_to_stance[i]:
                                        # found a complete stance
                                        complete_stances.append(stance_to_swing[j] - swing_to_stance[i])
                                        break
                                        
                            for i in range(len(stance_to_swing)):
                                for j in range(len(swing_to_stance)):
                                    if swing_to_stance[j] > stance_to_swing[i]:
                                        # found a complete swing
                                        complete_swings.append(swing_to_stance[j] - stance_to_swing[i])
                                        break
                            
                            if complete_stances:
                                stance_duration.append(np.mean(complete_stances) / fps)
                            if complete_swings:
                                swing_duration.append(np.mean(complete_swings) / fps)
                    
                    if stance_duration and swing_duration:
                        avg_stance = np.mean(stance_duration)
                        avg_swing = np.mean(swing_duration)
                        avg_duty_factor = avg_stance / (avg_stance + avg_swing)
                        
                        stance_durations.append(avg_stance)
                        swing_durations.append(avg_swing)
                        duty_factors.append(avg_duty_factor)
                    else:
                        # default values if we couldn't calculate
                        stance_durations.append(0.1)
                        swing_durations.append(0.05)
                        duty_factors.append(0.67)
                    
                    # asymmetry findings
                    n_feet = len(feet_forces)
                    if n_feet >= 2:
                        mid_idx = n_feet // 2
                        left_contacts = contacts[clip_frame, :mid_idx]
                        right_contacts = contacts[clip_frame, mid_idx:2*mid_idx]
                        asymmetry = np.mean(left_contacts) - np.mean(right_contacts)
                        asymmetry_values.append(asymmetry)
                    else:
                        asymmetry_values.append(0)
                    
                    # phase and frequency using Hilbert transform as in the paper & use the mean vertical force across all feet
                    mean_force = np.mean(feet_forces, axis=1)
                    
                    # window around center frame
                    window_start = max(0, clip_frame - half_window)
                    window_end = min(clip_data.shape[0], clip_frame + half_window)
                    window_force = mean_force[window_start:window_end]
                    
                    if len(window_force) >= 5:
                        # filter the signal
                        b, a = butter(4, [0.5, 10], fs=fps, btype='band')
                        try:
                            filtered_force = filtfilt(b, a, window_force)
                            
                            # hilbert transform
                            analytic_signal = hilbert(filtered_force)
                            
                            # phase at center
                            center_idx = clip_frame - window_start
                            if center_idx < len(analytic_signal):
                                phase = np.angle(analytic_signal[center_idx])
                                
                                # frequency
                                unwrapped_phase = np.unwrap(np.angle(analytic_signal))
                                if len(unwrapped_phase) > 1:
                                    freq = np.mean(np.diff(unwrapped_phase)) / (2.0 * np.pi) * fps
                                else:
                                    freq = 1.0
                                
                                phases.append(phase)
                                frequencies.append(freq)
                                
                                # store valid center frame and clip index
                                valid_center_frames.append(center_frame)
                                clip_indices.append(clip_idx)
                        except:
                            # skip if filtering fails
                            continue
                    else:
                        # skip if window is too small
                        continue
                except Exception as e:
                    # skip if processing fails
                    continue
    
    metrics = {
        'phases': np.array(phases) if phases else np.array([]),
        'frequencies': np.array(frequencies) if frequencies else np.array([]),
        'feet_in_stance': np.array(feet_in_stance) if feet_in_stance else np.array([]),
        'stance_durations': np.array(stance_durations) if stance_durations else np.array([]),
        'swing_durations': np.array(swing_durations) if swing_durations else np.array([]),
        'duty_factor': np.array(duty_factors) if duty_factors else np.array([]),
        'asymmetry': np.array(asymmetry_values) if asymmetry_values else np.array([]),
        'clip_indices': np.array(clip_indices) if clip_indices else np.array([]),
        'valid_center_frames': np.array(valid_center_frames) if valid_center_frames else np.array([])
    }
    
    print(f"Extracted gait metrics for {len(phases)} frames out of {len(center_frames)} requested frames")
    
    return metrics



def process_manifold_gait(int_data, jf_array, window_size=31, stride=1, n_samples=10000):
    """
    Main function matching the paper's methodology
    
    Parameters:
    - int_data: Intention data array
    - jf_array: Force data array
    - species: Species for feet indices
    - window_size: Size of time windows
    - stride: Step size between windows
    - n_samples: Number of samples to draw
    
    Returns:
    - Dictionary of results
    """
    print(f"Processing data with paper methods, window_size={window_size}")
    
    jf_array_reshape = jf_array.reshape(-1, jf_array.shape[-1])
    combined_intentions = np.hstack([int_data, jf_array_reshape])
    
    windows, center_frames = create_time_windows(int_data, window_size, stride)
    
    # sample diverse windows using PCA and clustering
    sampled_windows, sample_indices = sample_diverse_windows(
        windows, n_samples=min(n_samples, len(windows))
    )
    sampled_center_frames = center_frames[sample_indices]
    
    # extract gait metrics
    gait_metrics = estimate_gait_metrics(
        jf_array, sampled_center_frames, window_size
    )
    
    valid_indices = np.where(np.isin(sampled_center_frames, gait_metrics['valid_center_frames']))[0]
    
    if len(valid_indices) > 10:
        print(f"Using {len(valid_indices)} windows with valid gait metrics")
        windows_for_umap = sampled_windows[valid_indices]
    else:
        print("Using all sampled windows for UMAP")
        windows_for_umap = sampled_windows
    
    # UMAP embeddings
    embedded = apply_umap_to_intention_windows(
        windows_for_umap,
        n_components=3
        )
    
    trajectory_indices = identify_trajectory_indices(embedded, length=min(100, len(embedded)))
    
    if len(gait_metrics['phases']) > len(embedded):
        for key in gait_metrics:
            if isinstance(gait_metrics[key], np.ndarray):
                gait_metrics[key] = gait_metrics[key][:len(embedded)]
    elif len(gait_metrics['phases']) < len(embedded):
        embedded = embedded[:len(gait_metrics['phases'])]
    
    # analyze gait clusters
    n_clusters = min(4, len(embedded) // 50)
    n_clusters = max(2, n_clusters)
    cluster_labels = analyze_gait_cluster_distribution(
        embedded, gait_metrics, n_clusters
    )
    
    visualize_manifold(embedded, gait_metrics, trajectory_indices)
    visualize_additional_gait_metrics(embedded, gait_metrics)
    analyze_cyclic_structure(embedded, gait_metrics)
    create_multi_angle_view(embedded, gait_metrics, trajectory_indices)
    
    return {
        'embedded': embedded,
        'windows': windows_for_umap,
        'gait_metrics': gait_metrics,
        'trajectory_indices': trajectory_indices,
        'cluster_labels': cluster_labels
    }


def subsample_for_tda(embedded, max_points):
    if len(embedded) <= max_points:
        return embedded
    
    indices = np.random.choice(len(embedded), max_points, replace=False)
    return embedded[indices]


# def compute_persistence_diagrams(embedded, max_dim=2):
#     if len(embedded) < 2:
#         empty_result = {
#             'diagrams': [np.array([]), np.array([])],
#             'H0': np.array([]),
#             'H1': np.array([]),
#             'H2': None,
#             'H1_persistence': np.array([]),
#             'H1_sorted': np.array([]),
#             'H1_stats': {
#                 'mean': 0,
#                 'median': 0,
#                 'max': 0,
#                 'top_5': np.array([])
#             }
#         }
#         return empty_result
    
#     try:
#         diagrams = ripser.ripser(
#             embedded, 
#             maxdim=max_dim,
#             thresh=np.inf,
#             coeff=2,
#             do_cocycles=False,
#         )['dgms']
        
#         H0 = diagrams[0]
#         H1 = diagrams[1] if len(diagrams) > 1 else np.array([])
        
#         H2 = None
#         if max_dim >= 2 and len(diagrams) > 2:
#             H2 = diagrams[2]
        
#         if len(H1) > 0:
#             H1_persistence = H1[:, 1] - H1[:, 0]
#             H1_sorted_idx = np.argsort(-H1_persistence)
#             H1_sorted = H1[H1_sorted_idx]
#             H1_sorted_persistence = H1_persistence[H1_sorted_idx]
            
#             H1_stats = {
#                 'mean': np.mean(H1_persistence),
#                 'median': np.median(H1_persistence),
#                 'max': np.max(H1_persistence),
#                 'top_5': H1_sorted_persistence[:min(5, len(H1_sorted_persistence))]
#             }
#         else:
#             H1_persistence = np.array([])
#             H1_sorted = np.array([])
#             H1_stats = {
#                 'mean': 0,
#                 'median': 0,
#                 'max': 0,
#                 'top_5': np.array([])
#             }
        
#         results = {
#             'diagrams': diagrams,
#             'H0': H0,
#             'H1': H1,
#             'H2': H2,
#             'H1_persistence': H1_persistence,
#             'H1_sorted': H1_sorted,
#             'H1_stats': H1_stats
#         }
        
#         return results
#     except Exception as e:
#         empty_result = {
#             'diagrams': [np.array([]), np.array([])],
#             'H0': np.array([]),
#             'H1': np.array([]),
#             'H2': None,
#             'H1_persistence': np.array([]),
#             'H1_sorted': np.array([]),
#             'H1_stats': {
#                 'mean': 0,
#                 'median': 0,
#                 'max': 0,
#                 'top_5': np.array([])
#             }
#         }
#         return empty_result


def compute_persistence_diagrams(embedded, max_dim=2):
    if len(embedded) < 2:
        empty_result = {
            'diagrams': [np.array([]) for _ in range(max_dim + 1)],
            'H0': np.array([]),
            'H1': np.array([]),
            'H2': np.array([]),
            'H1_persistence': np.array([]),
            'H1_sorted': np.array([]),
            'H1_stats': {'mean': 0, 'median': 0, 'max': 0, 'top_5': np.array([])},
            'H2_persistence': np.array([]),
            'H2_sorted': np.array([]),
            'H2_stats': {'mean': 0, 'median': 0, 'max': 0, 'top_5': np.array([])}
        }
        return empty_result

    try:
        diagrams = ripser.ripser(
            embedded,
            maxdim=max_dim,
            thresh=np.inf,
            coeff=2,
            do_cocycles=False,
        )['dgms']

        H0 = diagrams[0]
        H1 = diagrams[1] if len(diagrams) > 1 else np.array([])
        H2 = diagrams[2] if max_dim >= 2 and len(diagrams) > 2 else np.array([])

        # Compute H1 stats
        H1_persistence = H1[:, 1] - H1[:, 0] if len(H1) > 0 else np.array([])
        H1_sorted_idx = np.argsort(-H1_persistence) if len(H1_persistence) > 0 else []
        H1_sorted = H1[H1_sorted_idx] if len(H1_sorted_idx) > 0 else np.array([])
        H1_sorted_persistence = H1_persistence[H1_sorted_idx] if len(H1_sorted_idx) > 0 else np.array([])
        H1_stats = {
            'mean': np.mean(H1_persistence) if len(H1_persistence) > 0 else 0,
            'median': np.median(H1_persistence) if len(H1_persistence) > 0 else 0,
            'max': np.max(H1_persistence) if len(H1_persistence) > 0 else 0,
            'top_5': H1_sorted_persistence[:5] if len(H1_sorted_persistence) >= 5 else H1_sorted_persistence
        }

        # Compute H2 stats
        H2_persistence = H2[:, 1] - H2[:, 0] if len(H2) > 0 else np.array([])
        H2_sorted_idx = np.argsort(-H2_persistence) if len(H2_persistence) > 0 else []
        H2_sorted = H2[H2_sorted_idx] if len(H2_sorted_idx) > 0 else np.array([])
        H2_sorted_persistence = H2_persistence[H2_sorted_idx] if len(H2_sorted_idx) > 0 else np.array([])
        H2_stats = {
            'mean': np.mean(H2_persistence) if len(H2_persistence) > 0 else 0,
            'median': np.median(H2_persistence) if len(H2_persistence) > 0 else 0,
            'max': np.max(H2_persistence) if len(H2_persistence) > 0 else 0,
            'top_5': H2_sorted_persistence[:5] if len(H2_sorted_persistence) >= 5 else H2_sorted_persistence
        }

        return {
            'diagrams': diagrams,
            'H0': H0,
            'H1': H1,
            'H2': H2,
            'H1_persistence': H1_persistence,
            'H1_sorted': H1_sorted,
            'H1_stats': H1_stats,
            'H2_persistence': H2_persistence,
            'H2_sorted': H2_sorted,
            'H2_stats': H2_stats
        }

    except Exception as e:
        print("Error in compute_persistence_diagrams:", e)
        return {
            'diagrams': [np.array([]) for _ in range(max_dim + 1)],
            'H0': np.array([]),
            'H1': np.array([]),
            'H2': np.array([]),
            'H1_persistence': np.array([]),
            'H1_sorted': np.array([]),
            'H1_stats': {'mean': 0, 'median': 0, 'max': 0, 'top_5': np.array([])},
            'H2_persistence': np.array([]),
            'H2_sorted': np.array([]),
            'H2_stats': {'mean': 0, 'median': 0, 'max': 0, 'top_5': np.array([])}
        }



def extract_persistent_features(diagrams, threshold=0.1):
    H1 = diagrams['H1']
    H1_persistence = diagrams['H1_persistence']
    
    persistent_H1_idx = np.where(H1_persistence > threshold)[0] if len(H1_persistence) > 0 else np.array([])
    persistent_H1 = H1[persistent_H1_idx] if len(persistent_H1_idx) > 0 else np.array([])
    
    persistent_H2 = None
    if diagrams['H2'] is not None:
        H2 = diagrams['H2']
        H2_persistence = H2[:, 1] - H2[:, 0]
        persistent_H2_idx = np.where(H2_persistence > threshold)[0]
        persistent_H2 = H2[persistent_H2_idx] if len(persistent_H2_idx) > 0 else None
    
    return {
        'H1': persistent_H1,
        'H2': persistent_H2,
        'H1_idx': persistent_H1_idx
    }


def landmark_subsample(points, n_landmarks=300):
    if len(points) <= n_landmarks:
        return points
    indices = np.random.choice(len(points), n_landmarks, replace=False)
    return points[indices]


def compute_topological_features(embedded, max_dim=1, n_landmarks=20, max_edge_length=0.5):
    if len(embedded) < 2:
        return {
            'diagram': [np.array([])],
            'landscape': np.zeros((1, 20)),
            'silhouette': np.zeros((1, 20)),
            'persistence_image': np.zeros((1, 100))
        }
    
    embedded_landmarks = landmark_subsample(embedded, n_landmarks=n_landmarks)
    
    try:
        # we use Gudhi for persistence computation
        rips_complex = gd.RipsComplex(points=embedded_landmarks, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim+1)
        simplex_tree.compute_persistence()
        
        persistence = simplex_tree.persistence_intervals_in_dimension
        diagram = [persistence(i) for i in range(max_dim+1)]
        
        # manual implementation of landscape computation
        if len(diagram) > 1 and len(diagram[1]) > 0:
            # manual landscape computation for H1
            h1_diagram = diagram[1]
            resolution = 20
            landscape_features = compute_landscape_features(h1_diagram, resolution=resolution)
            
            # manual silhouette computation for H1
            silhouette_features = compute_silhouette_features(h1_diagram, resolution=resolution)
            
            # manual persistence image computation
            pi_features = compute_persistence_image_features(h1_diagram, resolution=(10, 10))
        else:
            landscape_features = np.zeros((1, 20))
            silhouette_features = np.zeros((1, 20))
            pi_features = np.zeros((1, 100))
        
        return {
            'diagram': diagram,
            'landscape': landscape_features,
            'silhouette': silhouette_features,
            'persistence_image': pi_features
        }
    except Exception as e:
        return {
            'diagram': [np.array([])],
            'landscape': np.zeros((1, 20)),
            'silhouette': np.zeros((1, 20)),
            'persistence_image': np.zeros((1, 100))
        }


def compute_landscape_features(diagram, resolution=20, num_landscapes=5):
    """Compute persistence landscape features manually."""
    if len(diagram) == 0:
        return np.zeros((1, resolution))
    
    # get persistence pairs (birth, death)
    persistence_pairs = diagram
    
    grid = np.linspace(0, 1, resolution)
    
    landscapes = np.zeros((num_landscapes, resolution))
    
    for i, x in enumerate(grid):
        # compute all function values at this grid point
        values = []
        for birth, death in persistence_pairs:
            if birth <= x <= death:
                values.append(min(x - birth, death - x))
            else:
                values.append(0)
        
        values.sort(reverse=True)
        
        # assign to landscapes
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, i] = values[k]
    
    landscape_features = landscapes.reshape(1, -1)
    
    if landscape_features.shape[1] != resolution:
        pad_size = resolution - (landscape_features.shape[1] % resolution)
        landscape_features = np.pad(landscape_features, ((0, 0), (0, pad_size)), 'constant')
        landscape_features = landscape_features[:, :resolution]
    
    return landscape_features


def compute_silhouette_features(diagram, resolution=20, weight=lambda x: 1):
    """Compute persistence silhouette features manually."""
    if len(diagram) == 0:
        return np.zeros((1, resolution))
    
    # Get persistence pairs (birth, death)
    persistence_pairs = diagram
    
    # Compute persistence values
    persistences = np.array([death - birth for birth, death in persistence_pairs])
    
    # Define the grid
    grid = np.linspace(0, 1, resolution)
    
    # Initialize silhouette
    silhouette = np.zeros(resolution)
    
    # For each grid point
    for i, x in enumerate(grid):
        # Compute the silhouette value at this grid point
        val = 0
        total_weight = 0
        
        for j, (birth, death) in enumerate(persistence_pairs):
            if birth <= x <= death:
                w = weight(persistences[j])
                val += w * (min(x - birth, death - x) / persistences[j])
                total_weight += w
        
        if total_weight > 0:
            silhouette[i] = val / total_weight
    
    return silhouette.reshape(1, -1)

def compute_persistence_image_features(diagram, resolution=(10, 10), spread=0.1):
    """Compute persistence image features manually."""
    if len(diagram) == 0:
        return np.zeros((1, resolution[0] * resolution[1]))
    
    # persistence pairs (birth, death)
    persistence_pairs = diagram
    
    # convert to persistence-birth coordinates
    pers_birth = np.array([(death - birth, birth) for birth, death in persistence_pairs])
    
    x_grid = np.linspace(0, 1, resolution[0])
    y_grid = np.linspace(0, 1, resolution[1])
    
    image = np.zeros((resolution[0], resolution[1]))
    
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            val = 0
            
            for pers, birth in pers_birth:
                # Gaussian kernel
                val += pers * np.exp(-((x - pers)**2 + (y - birth)**2) / (2 * spread**2))
            
            image[i, j] = val
    
    return image.reshape(1, -1)


def identify_topological_regions(embedded, diagrams, persistent_features, gait_metrics):
    phases = gait_metrics.get('phases', np.array([]))
    frequencies = gait_metrics.get('frequencies', np.array([]))
    duty_factor = gait_metrics.get('duty_factor', np.array([]))
    
    persistent_H1 = persistent_features['H1']
    
    if len(persistent_H1) == 0 or len(embedded) == 0:
        return {
            'regions': [],
            'correlations': {}
        }
    
    regions = []
    correlations = {}
    
    for i, feature in enumerate(persistent_H1):
        birth, death = feature
        
        midpoint = (birth + death) / 2
        distances = np.abs(np.linalg.norm(embedded, axis=1) - midpoint)
        
        tolerance = (death - birth) * 0.2
        near_loop = distances < tolerance
        
        regions.append(near_loop)
        
        if np.sum(near_loop) > 5:
            if len(phases) > 0 and len(phases) >= np.sum(near_loop):
                phase_corr, p_phase = pearsonr(phases[near_loop], distances[near_loop])
            else:
                phase_corr, p_phase = 0, 1
                
            if len(frequencies) > 0 and len(frequencies) >= np.sum(near_loop):
                freq_corr, p_freq = pearsonr(frequencies[near_loop], distances[near_loop])
            else:
                freq_corr, p_freq = 0, 1
                
            if len(duty_factor) > 0 and len(duty_factor) >= np.sum(near_loop):
                duty_corr, p_duty = pearsonr(duty_factor[near_loop], distances[near_loop])
            else:
                duty_corr, p_duty = 0, 1
            
            correlations[f'loop_{i}'] = {
                'phase': (phase_corr, p_phase),
                'frequency': (freq_corr, p_freq),
                'duty_factor': (duty_corr, p_duty),
                'size': np.sum(near_loop),
                'birth': birth,
                'death': death,
                'persistence': death - birth
            }
    
    return {
        'regions': regions,
        'correlations': correlations
    }


def visualize_persistence_diagrams(diagrams, title="Persistence Diagrams"):
    plt.figure(figsize=(12, 8))
    try:
        plot_diagrams(diagrams['diagrams'], show=False)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        plt.text(0.5, 0.5, "Error plotting diagrams", ha='center', va='center')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()


def visualize_topological_correlations(tda_results, gait_metrics, embedded):
    if 'correlations' not in tda_results or len(tda_results['correlations']) == 0 or len(embedded) == 0:
        print("No correlations to visualize or empty embedded data")
        return
    
    correlations = tda_results['correlations']
    regions = tda_results['regions']
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(
        embedded[:, 0],
        embedded[:, 1],
        embedded[:, 2] if embedded.shape[1] > 2 else np.zeros(len(embedded)),
        c='lightgray',
        alpha=0.3,
        s=5
    )
    
    colors = plt.cm.tab10.colors
    for i, region in enumerate(regions):
        if i < len(correlations) and np.sum(region) > 0:
            ax1.scatter(
                embedded[region, 0],
                embedded[region, 1],
                embedded[region, 2] if embedded.shape[1] > 2 else np.zeros(np.sum(region)),
                c=[colors[i % len(colors)]],
                alpha=0.6,
                s=10,
                label=f"Loop {i}"
            )
    
    ax1.set_title("Topological Regions on Manifold", fontsize=14)
    ax1.legend()
    
    ax2 = fig.add_subplot(232)
    
    loop_ids = list(correlations.keys())
    gait_params = ['phase', 'frequency', 'duty_factor']
    
    if len(loop_ids) > 0:
        corr_matrix = np.zeros((len(loop_ids), len(gait_params)))
        
        for i, loop_id in enumerate(loop_ids):
            for j, param in enumerate(gait_params):
                corr_matrix[i, j] = correlations[loop_id][param][0]
        
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_xticks(np.arange(len(gait_params)))
        ax2.set_yticks(np.arange(len(loop_ids)))
        ax2.set_xticklabels(gait_params)
        ax2.set_yticklabels(loop_ids)
        plt.colorbar(im, ax=ax2, label="Correlation")
        
        for i in range(len(loop_ids)):
            for j in range(len(gait_params)):
                text = ax2.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", color="black")
    
    ax2.set_title("Correlation: Topological Features vs Gait Parameters", fontsize=14)
    
    ax3 = fig.add_subplot(233)
    
    if len(loop_ids) > 0:
        persistence_values = [correlations[loop_id]['persistence'] for loop_id in loop_ids]
        duty_corr_values = [abs(correlations[loop_id]['duty_factor'][0]) for loop_id in loop_ids]
        phase_corr_values = [abs(correlations[loop_id]['phase'][0]) for loop_id in loop_ids]
        freq_corr_values = [abs(correlations[loop_id]['frequency'][0]) for loop_id in loop_ids]
        
        ax3.scatter(persistence_values, duty_corr_values, label='Duty Factor', alpha=0.7, s=100)
        ax3.scatter(persistence_values, phase_corr_values, label='Phase', alpha=0.7, s=100)
        ax3.scatter(persistence_values, freq_corr_values, label='Frequency', alpha=0.7, s=100)
        
        ax3.set_xlabel('Persistence')
        ax3.set_ylabel('Absolute Correlation')
        ax3.set_title('Feature Persistence vs. Parameter Correlation', fontsize=14)
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    ax4 = fig.add_subplot(234)
    
    if len(loop_ids) > 0:
        sizes = [correlations[loop_id]['size'] for loop_id in loop_ids]
        
        bars = ax4.bar(loop_ids, sizes, color=[colors[i % len(colors)] for i in range(len(loop_ids))])
        
        ax4.set_xlabel('Topological Feature')
        ax4.set_ylabel('Number of Points')
        ax4.set_title('Size of Topological Regions', fontsize=14)
        
        if len(sizes) > 0:
            max_region_idx = np.argmax(sizes)
            max_region_id = loop_ids[max_region_idx]
            max_region = regions[max_region_idx]
            
            ax5 = fig.add_subplot(235)
            
            if len(gait_metrics.get('duty_factor', [])) > 0 and len(gait_metrics.get('frequencies', [])) > 0:
                ax5.hist(gait_metrics['duty_factor'][max_region], alpha=0.7, label='Duty Factor')
                ax5.hist(gait_metrics['frequencies'][max_region], alpha=0.7, label='Frequency')
                
                ax5.set_xlabel('Parameter Value')
                ax5.set_ylabel('Count')
                ax5.set_title(f'Parameter Distribution in {max_region_id}', fontsize=14)
                ax5.legend()
            
            ax6 = fig.add_subplot(236, projection='polar')
            
            if len(gait_metrics.get('phases', [])) > 0:
                for i, region in enumerate(regions):
                    if np.sum(region) > 5:
                        hist, bins = np.histogram(
                            gait_metrics['phases'][region], 
                            bins=16, 
                            range=(-np.pi, np.pi)
                        )
                        width = (bins[1] - bins[0])
                        ax6.bar(
                            bins[:-1], 
                            hist/np.sum(hist), 
                            width=width, 
                            alpha=0.5,
                            label=f"Loop {i}",
                            color=colors[i % len(colors)]
                        )
                
                ax6.set_title('Phase Distribution by Topological Region', fontsize=14)
                ax6.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def cluster_based_on_topology(embedded, tda_features):
    if len(embedded) < 2:
        return np.zeros(len(embedded), dtype=int)
        
    landscape_features = tda_features['landscape']
    
    if landscape_features.shape[0] == 1:
        point_features = np.tile(landscape_features[0], (len(embedded), 1))
    else:
        point_features = landscape_features
    
    # Make sure point_features has the right shape
    if point_features.shape[0] != embedded.shape[0]:
        point_features = np.tile(point_features[0], (embedded.shape[0], 1))
    
    combined_features = np.hstack([
        StandardScaler().fit_transform(embedded),
        StandardScaler().fit_transform(point_features)
    ])
    
    n_clusters = min(5, len(embedded) // 100)
    n_clusters = max(2, n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(combined_features)
    
    return labels


def manifold_with_tda_optimized(results, max_points):
    embedded = results['embedded']
    gait_metrics = results.get('gait_metrics', {})
    
    if len(embedded) < 2:
        print("Not enough points for TDA analysis")
        return None
    
    min_size = min(len(embedded), len(gait_metrics.get('phases', embedded)))
    embedded = embedded[:min_size]
    
    for key in gait_metrics:
        if isinstance(gait_metrics[key], np.ndarray) and len(gait_metrics[key]) > min_size:
            gait_metrics[key] = gait_metrics[key][:min_size]
    
    subsampled_embedded = subsample_for_tda(embedded, max_points=max_points)
    
    diagrams = compute_persistence_diagrams(
        subsampled_embedded, 
        max_dim=2
    )
    
    visualize_persistence_diagrams(diagrams, "Persistence Diagrams (Optimized)")
    
    diameter = np.max(np.linalg.norm(subsampled_embedded, axis=1)) if len(subsampled_embedded) > 0 else 0
    threshold = diameter * 0.2
    persistent_features = extract_persistent_features(diagrams, threshold)
    
    tda_features = compute_topological_features(subsampled_embedded)
    
    if len(subsampled_embedded) < len(embedded) and len(subsampled_embedded) > 0:
        nn = NearestNeighbors(n_neighbors=1).fit(subsampled_embedded)
        _, indices = nn.kneighbors(embedded)
        indices = indices.flatten()
    
    gait_metrics_subsampled = {k: v[:len(subsampled_embedded)] if isinstance(v, np.ndarray) else v 
                            for k, v in gait_metrics.items()}
    
    tda_results = identify_topological_regions(
        subsampled_embedded, diagrams, persistent_features, gait_metrics_subsampled
    )
    
    if tda_results:
        visualize_topological_correlations(tda_results, gait_metrics_subsampled, subsampled_embedded)
    
    full_tda_features = {
        'landscape': tda_features['landscape']
    }
    
    topo_clusters = cluster_based_on_topology(embedded, full_tda_features)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='3d')
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        embedded[:, 2] if embedded.shape[1] > 2 else np.zeros(len(embedded)),
        c=topo_clusters,
        cmap='tab10',
        alpha=0.6,
        s=5
    )
    ax.set_title("Manifold Clusters Based on Topology", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.tight_layout()
    plt.show()
    
    return {
        'diagrams': diagrams,
        'persistent_features': persistent_features,
        'tda_features': tda_features,
        'tda_results': tda_results,
        'topo_clusters': topo_clusters
    }


def sliding_window_tda(embedded, window_size=50, stride=10, max_points=500, gait_metrics=None):
    """Perform TDA on sliding windows of data."""
    n_frames = embedded.shape[0]
    n_windows = max(0, (n_frames - window_size) // stride + 1)
    
    if n_windows == 0:
        return []
    
    all_results = []
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        window_data = embedded[start_idx:end_idx]
        
        if len(window_data) > max_points:
            indices = np.random.choice(len(window_data), max_points, replace=False)
            window_data = window_data[indices]
        
        diagrams = compute_persistence_diagrams(window_data, max_dim=2)
        
        window_gait_metrics = None
        if gait_metrics is not None:
            window_gait_metrics = {
                key: value[start_idx:end_idx] if isinstance(value, np.ndarray) and len(value) >= end_idx else value
                for key, value in gait_metrics.items()
            }
        
        diameter = np.max(np.linalg.norm(window_data, axis=1)) if len(window_data) > 0 else 0
        threshold = diameter * 0.1
        persistent_features = extract_persistent_features(diagrams, threshold)
        
        tda_regions = None
        if window_gait_metrics is not None:
            try:
                tda_regions = identify_topological_regions(
                    window_data, diagrams, persistent_features, window_gait_metrics
                )
            except Exception as e:
                tda_regions = None
        
        window_result = {
            'window_idx': i,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'diagrams': diagrams,
            'persistent_features': persistent_features,
            'tda_regions': tda_regions
        }
        
        all_results.append(window_result)
    
    return all_results


def visualize_window_evolution(window_results):
    """Visualize the evolution of topological features across windows."""
    n_windows = len(window_results)
    if n_windows == 0:
        return
    
    window_indices = [r['window_idx'] for r in window_results]
    h1_means = [r['diagrams']['H1_stats']['mean'] for r in window_results]
    h1_maxes = [r['diagrams']['H1_stats']['max'] for r in window_results]
    h1_counts = [len(r['diagrams']['H1']) for r in window_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(window_indices, h1_means, 'o-', linewidth=2)
    axes[0, 0].set_title('Mean H1 Persistence Over Windows')
    axes[0, 0].set_xlabel('Window Index')
    axes[0, 0].set_ylabel('Mean Persistence')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    axes[0, 1].plot(window_indices, h1_maxes, 'o-', linewidth=2, color='orange')
    axes[0, 1].set_title('Max H1 Persistence Over Windows')
    axes[0, 1].set_xlabel('Window Index')
    axes[0, 1].set_ylabel('Max Persistence')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 0].plot(window_indices, h1_counts, 'o-', linewidth=2, color='green')
    axes[1, 0].set_title('Number of H1 Features Over Windows')
    axes[1, 0].set_xlabel('Window Index')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    try:
        max_features = max([len(r['diagrams'].get('H1_sorted', [])) for r in window_results])
        if max_features > 0:
            max_to_show = min(5, max_features)
            persistence_matrix = np.zeros((n_windows, max_to_show))
            
            for i, result in enumerate(window_results):
                sorted_persistence = result['diagrams'].get('H1_sorted', [])
                for j in range(min(max_to_show, len(sorted_persistence))):
                    if j < len(sorted_persistence) and sorted_persistence.shape[0] > 0:
                        persistence_matrix[i, j] = sorted_persistence[j, 1] - sorted_persistence[j, 0]
            
            im = axes[1, 1].imshow(persistence_matrix, aspect='auto', cmap='viridis')
            axes[1, 1].set_title('Top H1 Persistence Values Over Windows')
            axes[1, 1].set_xlabel('Top Features')
            axes[1, 1].set_ylabel('Window Index')
            plt.colorbar(im, ax=axes[1, 1], label='Persistence')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, "Error creating persistence heatmap", 
                      ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    try:
        n_diagrams_to_show = min(6, n_windows)
        indices_to_show = np.linspace(0, n_windows-1, n_diagrams_to_show, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices_to_show):
            if i < len(axes):
                result = window_results[idx]
                ax = axes[i]
                try:
                    plot_diagrams(result['diagrams']['diagrams'], show=False, ax=ax)
                    ax.set_title(f"Window {idx} (Frames {result['start_frame']}-{result['end_frame']})")
                except:
                    ax.text(0.5, 0.5, "No diagram data", ha='center', va='center')
                    ax.set_title(f"Window {idx} - No Data")
    except Exception as e:
        pass
    
    plt.tight_layout()
    plt.show()
    

def manual_plot_barcode(diagram, ax, title=""):
    """Plot a barcode diagram from birth-death pairs using matplotlib."""
    if len(diagram) == 0:
        ax.text(0.5, 0.5, "No features", ha='center', va='center')
        ax.set_title(title)
        return

    for i, (birth, death) in enumerate(diagram):
        ax.hlines(y=i, xmin=birth, xmax=death, linewidth=2)
        ax.plot([birth, death], [i, i], 'o', markersize=4)

    ax.set_ylim(-1, len(diagram))
    ax.set_xlim(np.min(diagram[:, 0]) - 0.1, np.max(diagram[:, 1]) + 0.1)
    ax.set_xlabel("Persistence (Birth → Death)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)


def visualize_h1_barcodes(window_results, n_show=6):
    """
    Show barcode plots of H1 persistence diagrams for a selected set of windows.
    """
    n_windows = len(window_results)
    if n_windows == 0:
        print("No window results to visualize.")
        return

    indices_to_show = np.linspace(0, n_windows - 1, min(n_show, n_windows), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices_to_show):
        result = window_results[idx]
        ax = axes[i]
        h1 = result['diagrams'].get('H1', np.array([]))
        if isinstance(h1, list):
            h1 = np.array(h1)
        if h1.size > 0:
            manual_plot_barcode(h1, ax, title=f"Barcode: Window {idx}")
        else:
            ax.text(0.5, 0.5, "No H1 features", ha='center', va='center')
            ax.set_title(f"Window {idx} - No H1")

    plt.tight_layout()
    plt.show()


def process_intention_tda(int_rodent, jf_rodent, window_size=50, stride=10, max_points=1000, gait_metrics=None, embedding_option='combine'):
    """Process intention data directly using TDA with rolling windows and manifold analysis."""
    if len(int_rodent.shape) > 2:
        int_rodent = int_rodent.reshape(-1, int_rodent.shape[-1])
    
    if len(jf_rodent.shape) > 2:
        jf_rodent = jf_rodent.reshape(-1, jf_rodent.shape[-1])
    
    if embedding_option == 'combine':
        print('Using combined embeddings')
        combined_intentions = np.hstack([int_rodent, jf_rodent])
        
    if embedding_option == 'intention':
        print('Using intention embeddings')
        combined_intentions = int_rodent
        
    if embedding_option == 'force':
        print('Using force embeddings')
        combined_intentions = jf_rodent
    
    if combined_intentions.shape[0] < window_size:
        raise ValueError(f"Not enough data points. Got {combined_intentions.shape[0]} points but window_size is {window_size}")
    
    reducer = umap.UMAP(n_components=3, min_dist=0.1, n_neighbors=min(30, combined_intentions.shape[0]-1), random_state=42)
    embedded = reducer.fit_transform(combined_intentions)
    
    all_results = {
        'embedded': embedded,
        'gait_metrics': gait_metrics,
        'window_results': [],
        'global_tda': None
    }
    
    # if combined_intentions.shape[0] > max_points:
    #     print(f"Subsampling from {combined_intentions.shape[0]} to {max_points} for PHATE")
    #     indices = np.random.choice(combined_intentions.shape[0], max_points, replace=False)
    #     combined_intentions_subsampled = combined_intentions[indices]
    #     if gait_metrics is not None:
    #         gait_metrics = {k: v[indices] if isinstance(v, np.ndarray) else v for k, v in gait_metrics.items()}
    # else:
    #     combined_intentions_subsampled = combined_intentions
    
    if embedded.shape[0] > 1:
        # give you tda upon the UMAP embedded manifold
        global_tda = manifold_with_tda_optimized({'embedded': embedded, 'gait_metrics': gait_metrics}, min(max_points, embedded.shape[0]))
        all_results['global_tda'] = global_tda
    else:
        all_results['global_tda'] = None
    
    # give you tda upon the intention/force/combined ambient space with isomap
    window_results = sliding_window_tda(combined_intentions, window_size=window_size, stride=stride, max_points=max_points, gait_metrics=gait_metrics)
    all_results['window_results'] = window_results
    
    if len(window_results) > 0:
        try:
            visualize_window_evolution(window_results)
            visualize_h1_barcodes(window_results)
        except Exception as e:
            pass
    
    return all_results


def analyze_intentions_with_tda(int_rodent, jf_rodent, window_size=50, stride=10, max_points=1000, gait_metrics=None, embedding_option='combine'):
    """Complete function to analyze intention data using TDA and manifold learning."""
    try:
        int_rodent_reshaped = int_rodent.squeeze() if len(int_rodent.shape) > 2 else int_rodent
        jf_rodent_reshaped = jf_rodent.squeeze() if len(jf_rodent.shape) > 2 else jf_rodent
        
        all_results = process_intention_tda(
            int_rodent_reshaped, 
            jf_rodent_reshaped, 
            window_size=window_size, 
            stride=stride, 
            max_points=max_points, 
            gait_metrics=gait_metrics,
            embedding_option=embedding_option
        )
        
        return all_results
    
    except Exception as e:
        print(f"Error in analyze_intentions_with_tda: {str(e)}")
        return None


def visualize_window_manifolds(all_results):
    """Create a grid of manifold plots for selected windows to see evolution."""
    n_windows = len(all_results['window_results'])
    if n_windows == 0:
        return
        
    n_to_show = min(9, n_windows)
    indices_to_show = np.linspace(0, n_windows-1, n_to_show, dtype=int)
    
    rows = int(np.ceil(np.sqrt(n_to_show)))
    cols = int(np.ceil(n_to_show / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), subplot_kw={'projection': '3d'})
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, idx in enumerate(indices_to_show):
        if i < len(axes):
            window = all_results['window_results'][idx]
            ax = axes[i]
            
            start_frame = window['start_frame']
            end_frame = window['end_frame']
            embed = all_results['embedded'][start_frame:end_frame]
            
            if len(embed) > 0:
                ax.scatter(
                    embed[:, 0],
                    embed[:, 1],
                    embed[:, 2] if embed.shape[1] > 2 else np.zeros(len(embed)),
                    alpha=0.7,
                    s=10,
                    c=np.arange(len(embed)),
                    cmap='viridis'
                )
                
                ax.set_title(f"Window {idx} (Frames {start_frame}-{end_frame})")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.set_zlabel("UMAP 3")
                
                if window.get('tda_regions') is not None and 'regions' in window['tda_regions']:
                    for j, region in enumerate(window['tda_regions']['regions']):
                        if np.sum(region) > 5:
                            color = plt.cm.tab10.colors[j % 10]
                            region_pts = np.where(region)[0]
                            if len(region_pts) > 0 and max(region_pts) < len(embed):
                                ax.scatter(
                                    embed[region, 0],
                                    embed[region, 1],
                                    embed[region, 2] if embed.shape[1] > 2 else np.zeros(np.sum(region)),
                                    color=color,
                                    alpha=0.9,
                                    s=20,
                                    label=f"Loop {j}"
                                )
                    
                    ax.legend(loc='upper right', fontsize='small')
    
    for i in range(n_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_window_with_global(all_results, window_idx=None):
    """Compare a specific window's TDA results with the global TDA results."""
    if all_results['global_tda'] is None:
        print("No global TDA results available")
        return
        
    if window_idx is None:
        if len(all_results['window_results']) > 0:
            window_idx = len(all_results['window_results']) // 2  # Use middle window if not specified
        else:
            print("No window results available")
            return
    
    if window_idx >= len(all_results['window_results']):
        print(f"Error: Window index {window_idx} out of range.")
        return
    
    window = all_results['window_results'][window_idx]
    global_tda = all_results['global_tda']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    try:
        plot_diagrams(global_tda['diagrams']['diagrams'], show=False, ax=axes[0, 0])
        axes[0, 0].set_title('Global Persistence Diagram')
    except:
        axes[0, 0].text(0.5, 0.5, "Error plotting global diagram", ha='center', va='center')
        axes[0, 0].set_title('Global Persistence Diagram - Error')
    
    try:
        plot_diagrams(window['diagrams']['diagrams'], show=False, ax=axes[0, 1])
        axes[0, 1].set_title(f'Window {window_idx} Persistence Diagram')
    except:
        axes[0, 1].text(0.5, 0.5, "Error plotting window diagram", ha='center', va='center')
        axes[0, 1].set_title(f'Window {window_idx} Persistence Diagram - Error')
    
    axes[1, 0].scatter(
        all_results['embedded'][:, 0],
        all_results['embedded'][:, 1],
        c='lightgray',
        alpha=0.3,
        s=5
    )
    
    start_frame = window['start_frame']
    end_frame = window['end_frame']
    if start_frame < len(all_results['embedded']) and end_frame <= len(all_results['embedded']):
        axes[1, 0].scatter(
            all_results['embedded'][start_frame:end_frame, 0],
            all_results['embedded'][start_frame:end_frame, 1],
            c='red',
            alpha=0.7,
            s=10
        )
    axes[1, 0].set_title('Global Manifold with Window Highlighted')
    
    window_pts = all_results['embedded'][start_frame:end_frame]
    if len(window_pts) > 0:
        axes[1, 1].scatter(
            window_pts[:, 0],
            window_pts[:, 1],
            c='blue',
            alpha=0.7,
            s=10
        )
    axes[1, 1].set_title(f'Window {window_idx} Local Manifold')
    
    plt.tight_layout()
    plt.show()
    
    global_h1_stats = global_tda['diagrams']['H1_stats']
    window_h1_stats = window['diagrams']['H1_stats']
    
    print("\nComparison of TDA Statistics:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Global':<15} {'Window':<15}")
    print("-" * 50)
    
    for stat in ['mean', 'median', 'max']:
        print(f"{stat:<20} {global_h1_stats[stat]:<15.4f} {window_h1_stats[stat]:<15.4f}")
    
    print("-" * 50)
    print(f"{'Top 5 Global':<20} {'Value':<15}")
    for i, val in enumerate(global_h1_stats['top_5']):
        print(f"{f'  #{i+1}':<20} {val:<15.4f}")
    
    print("-" * 50)
    print(f"{'Top 5 Window':<20} {'Value':<15}")
    for i, val in enumerate(window_h1_stats['top_5']):
        print(f"{f'  #{i+1}':<20} {val:<15.4f}")


def analyze_intention_structure_over_time(all_results):
    """Analyze how the structure of the intention manifold changes over time."""
    n_windows = len(all_results['window_results'])
    if n_windows < 2:
        print("Not enough windows for temporal analysis")
        return
    
    window_indices = [r['window_idx'] for r in all_results['window_results']]
    start_frames = [r['start_frame'] for r in all_results['window_results']]
    
    # wasserstein distances between consecutive window diagrams
    window_tda_distances = []
    
    for i in range(1, n_windows):
        prev_window = all_results['window_results'][i-1]
        curr_window = all_results['window_results'][i]
        
        prev_diagram = prev_window['diagrams']['H1']
        curr_diagram = curr_window['diagrams']['H1']
        
        if len(prev_diagram) > 0 and len(curr_diagram) > 0:
            try:
                wass_dist = wasserstein(prev_diagram, curr_diagram)
            except:
                wass_dist = 0.0
        else:
            wass_dist = 0.0
        
        window_tda_distances.append(wass_dist)
    
    plt.figure(figsize=(12, 6))
    plt.plot(window_indices[1:], window_tda_distances, 'o-', linewidth=2)
    plt.title('Topological Distance Between Consecutive Windows')
    plt.xlabel('Window Index')
    plt.ylabel('Wasserstein Distance Between H1 Diagrams')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    
    h1_counts = [len(r['diagrams']['H1']) for r in all_results['window_results']]
    plt.plot(start_frames, h1_counts, 'o-', label='Number of H1 Features')
    
    h1_means = [r['diagrams']['H1_stats']['mean'] for r in all_results['window_results']]
    plt.plot(start_frames, h1_means, 'o-', label='Mean H1 Persistence')
    
    plt.title('Evolution of Topological Features Over Time')
    plt.xlabel('Start Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# def compare_tda_across_behaviors(behavior_data, window_size=50, max_points=300):
#     """Compare TDA features across different behavior types."""
#     behavior_tda_results = {}
#     all_h1_means = []
#     all_h1_maxes = []
#     all_behavior_types = []
    
#     for behavior_type, data in behavior_data.items():
#         int_data, jf_data = data['int'], data['jf']
        
#         if len(int_data.shape) > 2:
#             int_data = int_data.reshape(-1, int_data.shape[-1])
#         if len(jf_data.shape) > 2:
#             jf_data = jf_data.reshape(-1, jf_data.shape[-1])
        
#         combined = np.hstack([int_data, jf_data])
        
#         try:
#             reducer = umap.UMAP(n_components=3, min_dist=0.1, n_neighbors=min(30, combined.shape[0]-1), random_state=42)
#             embedded = combined #reducer.fit_transform(combined)
            
#             subsampled = subsample_for_tda(embedded, max_points=max_points)
            
#             diagrams = compute_persistence_diagrams(subsampled, max_dim=2)
            
#             h1_mean = diagrams['H1_stats']['mean']
#             h1_max = diagrams['H1_stats']['max']
            
#             all_h1_means.append(h1_mean)
#             all_h1_maxes.append(h1_max)
#             all_behavior_types.append(behavior_type)
            
#             behavior_tda_results[behavior_type] = {
#                 'diagrams': diagrams,
#                 'embedded': embedded,
#                 'stats': diagrams['H1_stats']
#             }
            
#             plt.figure(figsize=(12, 8))
#             plot_diagrams(diagrams['diagrams'], show=False)
#             plt.title(f"Persistence Diagram: {behavior_type}", fontsize=16)
#             plt.tight_layout()
#             plt.show()
#         except Exception as e:
#             print(f"Error processing {behavior_type}: {str(e)}")
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
#     # giev mean persistence
#     ax1.bar(all_behavior_types, all_h1_means, color='skyblue', alpha=0.7)
#     ax1.set_title('Mean H1 Persistence by Behavior Type')
#     ax1.set_xlabel('Behavior Type')
#     ax1.set_ylabel('Mean Persistence')
    
#     # give max persistence
#     ax2.bar(all_behavior_types, all_h1_maxes, color='salmon', alpha=0.7)
#     ax2.set_title('Max H1 Persistence by Behavior Type')
#     ax2.set_xlabel('Behavior Type')
#     ax2.set_ylabel('Max Persistence')
    
#     plt.tight_layout()
#     plt.show()
    
#     # when we have more than one behavior, compare manifolds
#     if len(behavior_data) > 1:
#         n_behaviors = len(behavior_data)
#         fig = plt.figure(figsize=(5*n_behaviors, 5))
        
#         for i, behavior_type in enumerate(behavior_tda_results.keys()):
#             ax = fig.add_subplot(1, n_behaviors, i+1, projection='3d')
#             embedded = behavior_tda_results[behavior_type]['embedded']
#             ax.scatter(
#                 embedded[:, 0], 
#                 embedded[:, 1], 
#                 embedded[:, 2] if embedded.shape[1] > 2 else np.zeros(len(embedded)),
#                 alpha=0.7,
#                 s=5
#             )
#             ax.set_title(f"{behavior_type} Manifold")
#             ax.set_xlabel("UMAP 1")
#             ax.set_ylabel("UMAP 2")
#             ax.set_zlabel("UMAP 3")
        
#         plt.tight_layout()
#         plt.show()
    
#     return behavior_tda_results


def compare_tda_across_behaviors(behavior_data, max_points=300, embedding_option='combine'):
    """Compare TDA features across different behavior types."""
    import matplotlib.pyplot as plt
    from persim import plot_diagrams

    behavior_tda_results = {}
    all_behavior_types = []

    all_h1_means, all_h1_maxes = [], []
    all_h2_means, all_h2_maxes = [], []

    for behavior_type, data in behavior_data.items():
        int_data, jf_data = data['int'], data['jf']

        if len(int_data.shape) > 2:
            int_data = int_data.reshape(-1, int_data.shape[-1])
        if len(jf_data.shape) > 2:
            jf_data = jf_data.reshape(-1, jf_data.shape[-1])

        if embedding_option == 'combine':
            print('Using combined embeddings')
            combined = np.hstack([int_data, jf_data])
            
        if embedding_option == 'intention':
            print('Using intention embeddings')
            combined = int_data
            
        if embedding_option == 'force':
            print('Using force embeddings')
            combined = jf_data

        try:
            reducer = umap.UMAP(n_components=3, min_dist=0.1, n_neighbors=min(30, combined.shape[0]-1), random_state=42)
            embedded = reducer.fit_transform(combined)

            subsampled = subsample_for_tda(combined, max_points=max_points)
            diagrams = compute_persistence_diagrams(subsampled, max_dim=2)

            h1_mean = diagrams['H1_stats']['mean']
            h1_max = diagrams['H1_stats']['max']
            h2_mean = diagrams['H2_stats']['mean']
            h2_max = diagrams['H2_stats']['max']

            all_h1_means.append(h1_mean)
            all_h1_maxes.append(h1_max)
            all_h2_means.append(h2_mean)
            all_h2_maxes.append(h2_max)
            all_behavior_types.append(behavior_type)

            behavior_tda_results[behavior_type] = {
                'diagrams': diagrams,
                'embedded': embedded,
                'H1_stats': diagrams['H1_stats'],
                'H2_stats': diagrams['H2_stats']
            }

            # Show persistence diagram
            plt.figure(figsize=(12, 8))
            plot_diagrams(diagrams['diagrams'], show=False)
            plt.title(f"Persistence Diagram: {behavior_type}", fontsize=16)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing {behavior_type}: {str(e)}")

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    axs[0, 0].bar(all_behavior_types, all_h1_means, color='skyblue', alpha=0.7)
    axs[0, 0].set_title('Mean H1 Persistence by Behavior Type')
    axs[0, 0].set_ylabel('Mean H1')

    axs[0, 1].bar(all_behavior_types, all_h1_maxes, color='salmon', alpha=0.7)
    axs[0, 1].set_title('Max H1 Persistence by Behavior Type')
    axs[0, 1].set_ylabel('Max H1')

    axs[1, 0].bar(all_behavior_types, all_h2_means, color='lightgreen', alpha=0.7)
    axs[1, 0].set_title('Mean H2 Persistence by Behavior Type')
    axs[1, 0].set_ylabel('Mean H2')

    axs[1, 1].bar(all_behavior_types, all_h2_maxes, color='orange', alpha=0.7)
    axs[1, 1].set_title('Max H2 Persistence by Behavior Type')
    axs[1, 1].set_ylabel('Max H2')

    for ax in axs.flatten():
        ax.set_xlabel('Behavior Type')
        ax.set_xticks(range(len(all_behavior_types)))
        ax.set_xticklabels(all_behavior_types, rotation=45)

    plt.tight_layout()
    plt.show()

    if len(behavior_data) > 1:
        n_behaviors = len(behavior_data)
        fig = plt.figure(figsize=(5 * n_behaviors, 5))

        for i, behavior_type in enumerate(behavior_tda_results.keys()):
            ax = fig.add_subplot(1, n_behaviors, i + 1, projection='3d')
            embedded = behavior_tda_results[behavior_type]['embedded']
            ax.scatter(
                embedded[:, 0],
                embedded[:, 1],
                embedded[:, 2] if embedded.shape[1] > 2 else np.zeros(len(embedded)),
                alpha=0.7,
                s=5
            )
            ax.set_title(f"{behavior_type} Manifold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_zlabel("UMAP 3")

        plt.tight_layout()
        plt.show()

    return behavior_tda_results
