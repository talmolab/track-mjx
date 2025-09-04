import pandas as pd
import numpy as np
from scipy import signal
import jax.numpy as jp
import logging
from typing import List, Tuple, Dict, Any

class EMGProcessor:
    def __init__(
        self, 
        muscle_configs: List[Tuple], 
        trial_csv_path: str, 
        fs: int = 30000, 
        target_fs: int = 400,
        target_samples: int = 200
    ):
        """
        Load and process EMG data for muscle activation comparison.
        
        Args:
            muscle_configs: List of tuples (actuator_idx, muscle_name, emg_path, emg_label)
            trial_csv_path: Path to CSV with trial start/end indices
            fs: Original sampling frequency in Hz
            target_fs: Target sampling frequency in Hz
            target_samples: Number of samples to extract for each trial
        """
        # Load trial data
        logging.info(f"Loading EMG trial data from {trial_csv_path}")
        self.df = pd.read_csv(trial_csv_path)
        
        # Create trial mask for valid trials
        self.trial_mask = (
            (self.df['start'].notna()) & 
            (self.df['end'].notna()) & 
            (self.df['start'] > 0) & 
            (self.df['end'] > 0) &
            (self.df['start'] != self.df['end'])
        )
        
        # Get valid trials
        self.valid_trials_df = self.df[self.trial_mask].copy()
        self.valid_trial_indices = self.valid_trials_df.index.tolist()
        
        # Set up filter parameters
        self.fs = fs
        self.target_fs = target_fs
        self.highpass_cutoff = 20
        self.lowpass_cutoff = 1000
        self.envelope_cutoff = 50
        self.filter_order = 4
        self.downsample_factor = fs // target_fs
        self.target_samples = target_samples
        
        logging.info(f"Found {len(self.valid_trials_df)} valid trials out of {len(self.df)} total trials")
        
        # Process EMG for each muscle
        self.muscle_emgs = {}
        for actuator_idx, muscle_name, emg_path, emg_label in muscle_configs:
            logging.info(f"Processing EMG for {muscle_name} ({emg_label}) at actuator {actuator_idx}")
            self.muscle_emgs[actuator_idx] = self.process_emg_data(emg_path)
            
        logging.info(f"EMG data loaded for {len(self.muscle_emgs)} muscles across {len(self.valid_trials_df)} valid trials")
    
    def process_emg_data(self, emg_file_path: str) -> List[np.ndarray]:
        """Process EMG data and return normalized envelope"""
        # Load EMG data
        emg_data = pd.read_csv(emg_file_path, header=None)
        logging.info(f"  Loaded EMG data: {len(emg_data)} recordings")
        
        # Process only valid trials using the trial mask
        reach_envelopes_downsampled = []
        
        for i, (idx, row) in enumerate(self.valid_trials_df.iterrows()):
            trial_num = idx
            
            # Convert reach start index from 200Hz frames to 30000Hz samples
            emg_reach_start = int(1/200 * row['start'] * self.fs)
            emg_reach_end = emg_reach_start + 15000  # 0.5 seconds (at 30kHz)
            
            # Get EMG data for this specific trial
            if trial_num < len(emg_data):
                trial_emg = emg_data.iloc[trial_num, :].values
                
                # Process EMG signal
                b, a = signal.butter(self.filter_order, [self.highpass_cutoff, self.lowpass_cutoff], 
                                    btype='bandpass', fs=self.fs)
                filtered_emg = signal.filtfilt(b, a, trial_emg)
                
                rectified_emg = np.abs(filtered_emg)
                
                b_env, a_env = signal.butter(self.filter_order, self.envelope_cutoff, 
                                           btype='lowpass', fs=self.fs)
                emg_envelope = signal.filtfilt(b_env, a_env, rectified_emg)
                
                # Extract envelope during reach period
                if emg_reach_start < len(emg_envelope) and emg_reach_end <= len(emg_envelope):
                    reach_envelope = emg_envelope[emg_reach_start:emg_reach_end]
                    
                    if len(reach_envelope) > 0:
                        # Downsample by averaging chunks
                        n_samples = len(reach_envelope)
                        n_chunks = n_samples // self.downsample_factor
                        
                        # Trim to exact multiple of downsample_factor
                        trimmed_envelope = reach_envelope[:n_chunks * self.downsample_factor]
                        
                        # Reshape and average
                        reshaped = trimmed_envelope.reshape(n_chunks, self.downsample_factor)
                        downsampled_envelope = np.mean(reshaped, axis=1)
                        
                        # Only keep if we have the target number of samples
                        if len(downsampled_envelope) >= self.target_samples:
                            downsampled_envelope = downsampled_envelope[:self.target_samples]
                            
                            # Normalize the envelope to [0, 1]
                            if np.max(downsampled_envelope) > 0:
                                downsampled_envelope = downsampled_envelope / np.max(downsampled_envelope)
                            
                            reach_envelopes_downsampled.append(downsampled_envelope)
            else:
                logging.warning(f"  Trial {trial_num}: EMG data index out of range")
        
        logging.info(f"  Processed {len(reach_envelopes_downsampled)} valid EMG sequences")
        return reach_envelopes_downsampled
    
    def get_emg_arrays_for_jax(self, clip_idx: int, frame_idx: int) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """
        Get EMG data in JAX-compatible array format for a specific clip and frame.
        
        This function runs outside of JAX compilation, so we can use Python control flow.
        
        Args:
            clip_idx: Index of the clip to get EMG data for
            frame_idx: Current frame index in the clip
            
        Returns:
            Tuple containing:
            - actuator_indices: Array of actuator indices that have EMG data
            - emg_values: Array of EMG values for those actuators at the current frame
            - valid_mask: Binary mask indicating valid comparisons (1=valid, 0=invalid)
        """
        # Pre-allocate lists to collect data
        actuator_indices = []
        emg_values = []
        valid_mask = []
        
        # For each muscle we're tracking
        for actuator_idx, emg_list in self.muscle_emgs.items():
            # Check if we have EMG data for this clip
            if clip_idx < len(emg_list):
                emg_array = emg_list[clip_idx]
                
                # Check if we have data for this frame
                if frame_idx < len(emg_array):
                    actuator_indices.append(actuator_idx)
                    emg_values.append(float(emg_array[frame_idx]))
                    valid_mask.append(1.0)  # Valid comparison
                else:
                    # Include the muscle but mark as invalid
                    actuator_indices.append(actuator_idx)
                    emg_values.append(0.0)  # Dummy value
                    valid_mask.append(0.0)  # Invalid comparison
        
        # If no data, return empty arrays with correct types
        if not actuator_indices:
            return jp.array([], dtype=jp.int32), jp.array([]), jp.array([])
        
        # Convert to JAX arrays
        return (
            jp.array(actuator_indices, dtype=jp.int32),
            jp.array(emg_values, dtype=jp.float32),
            jp.array(valid_mask, dtype=jp.float32)
        )
    
    def prepare_static_arrays(self):
        """Create static arrays for JAX compatibility"""
        actuator_indices = []
        emg_values = []
        valid_masks = []
        
        # Process EMG data into static arrays
        for actuator_idx, emg_list in self.muscle_emgs.items():
            actuator_indices.append(actuator_idx)
            
            # Use average EMG value across all clips and frames
            all_values = []
            for clip in emg_list:
                all_values.extend(clip)
            
            if all_values:
                avg_value = float(np.mean(all_values))
                emg_values.append(avg_value)
                valid_masks.append(1.0)
            else:
                emg_values.append(0.0)
                valid_masks.append(0.0)
        
        # Convert to JAX arrays
        return (
            jp.array(actuator_indices, dtype=jp.int32),
            jp.array(emg_values, dtype=jp.float32), 
            jp.array(valid_masks, dtype=jp.float32)
        )