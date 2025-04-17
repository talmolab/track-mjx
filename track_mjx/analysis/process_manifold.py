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


def identify_feet_indices(species):
    """
    Return indices of bodies that represent feet for different species
    Replace with your actual feet indices
    """
    if species == 'rodent':
        return [12, 13, 16, 17, 60, 61, 65, 66] 
    elif species == 'fly':
        return [30, 34, 38, 42, 46, 50]
    else:
        raise ValueError("Unknown species")


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


def estimate_gait_metrics(cfrc, feet_indices, center_frames, window_size=31, fps=30):
    """
    Extract gait metrics from force data using methods from the paper
    
    Parameters:
    - cfrc: Array of shape (n_clips, n_frames, n_bodies, 6) containing force data
    - feet_indices: Indices of bodies representing feet
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
            
            half_window = window_size // 2
            if clip_frame >= half_window and clip_frame < clip_data.shape[0] - half_window:
                try:
                    # extract vertical forces for feet
                    feet_forces = clip_data[:, feet_indices, 2]  # z-component
                    
                    # detect contacts (threshold force indicating stance)
                    contact_threshold = 0.1
                    contacts = feet_forces > contact_threshold
                    
                    # Get number of feet in stance at center frame
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
                    
                    # asymmetry
                    n_feet = len(feet_indices)
                    if n_feet >= 2:
                        mid_idx = n_feet // 2
                        left_contacts = contacts[clip_frame, :mid_idx]
                        right_contacts = contacts[clip_frame, mid_idx:2*mid_idx]
                        asymmetry = np.mean(left_contacts) - np.mean(right_contacts)
                        asymmetry_values.append(asymmetry)
                    else:
                        asymmetry_values.append(0)
                    
                    # phase and frequency using Hilbert transform as in the paper
                    # use the mean vertical force across all feet
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
                            
                            # Hilbert transform
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



def process_manifold_gait(int_data, jf_rodent, species='rodent', window_size=31, stride=1, n_samples=10000):
    """
    Main function matching the paper's methodology
    
    Parameters:
    - int_data: Intention data array
    - jf_rodent: Force data array
    - species: Species for feet indices
    - window_size: Size of time windows
    - stride: Step size between windows
    - n_samples: Number of samples to draw
    
    Returns:
    - Dictionary of results
    """
    print(f"Processing data with paper methods, window_size={window_size}")
    
    feet_indices = identify_feet_indices(species)
    windows, center_frames = create_time_windows(int_data, window_size, stride)
    
    # sample diverse windows using PCA and clustering
    sampled_windows, sample_indices = sample_diverse_windows(
        windows, n_samples=min(n_samples, len(windows))
    )
    sampled_center_frames = center_frames[sample_indices]
    
    # extract gait metrics
    gait_metrics = estimate_gait_metrics(
        jf_rodent, feet_indices, sampled_center_frames, window_size
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