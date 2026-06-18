"""
Post-processing module for 3D pose data.

Applies temporal smoothing, bone-length constraints, and interpolation
to raw 3D pose trajectories.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


# Anatomical bone length constraints (in meters)
# Based on average human proportions
BONE_LENGTH_CONSTRAINTS = {
    # Format: (joint1_idx, joint2_idx): (min_length, max_length)
    # Using Human3.6M/COCO joint ordering (16 joints)
    # Approximate average human proportions
    (0, 1): (0.15, 0.35),    # Hip center to spine/thorax
    (1, 2): (0.15, 0.35),    # Spine to neck
    (2, 3): (0.15, 0.30),    # Neck to head
    (2, 4): (0.10, 0.25),    # Neck to left shoulder
    (4, 5): (0.20, 0.40),    # Left shoulder to left elbow
    (5, 6): (0.20, 0.40),    # Left elbow to left wrist
    (2, 7): (0.10, 0.25),    # Neck to right shoulder
    (7, 8): (0.20, 0.40),    # Right shoulder to right elbow
    (8, 9): (0.20, 0.40),    # Right elbow to right wrist
    (0, 10): (0.10, 0.25),   # Hip center to left hip
    (10, 11): (0.35, 0.55),  # Left hip to left knee
    (11, 12): (0.35, 0.55),  # Left knee to left ankle
    (0, 13): (0.10, 0.25),   # Hip center to right hip
    (13, 14): (0.35, 0.55),  # Right hip to right knee
    (14, 15): (0.35, 0.55),  # Right knee to right ankle
}


def apply_butterworth_filter(
    data: np.ndarray,
    cutoff: float = 3.0,
    fs: float = 30.0,
    order: int = 2
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to remove high-frequency noise.
    
    Args:
        data: Input pose data with shape (frames, joints, 3) or (frames, features)
        cutoff: Low-pass filter cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered data with same shape as input
    
    Raises:
        ValueError: If input has fewer than 2 frames (filtering requires time series)
    """
    # filtfilt requires at least 3 * filter order frames (padlen = 3 * ntaps where ntaps = max(len(a), len(b)))
    # For a butterworth filter of order 2, we need at least 3 * 3 = 9 frames
    min_frames_required = max(3 * (order + 1), 10)
    
    if data.shape[0] < min_frames_required:
        # Cannot apply filter to insufficient data
        return data.copy()
    
    # Handle NaN values by filtering each trajectory independently
    filtered_data = data.copy()
    
    # Compute filter coefficients
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Ensure cutoff is valid (must be < 1.0)
    if normal_cutoff >= 1.0:
        raise ValueError(f"Cutoff frequency {cutoff} Hz must be less than Nyquist frequency {nyq} Hz")
    
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    
    # Apply filter along time axis (axis=0)
    # Handle multi-dimensional data
    if data.ndim == 3:  # (frames, joints, coords)
        for joint_idx in range(data.shape[1]):
            for coord_idx in range(data.shape[2]):
                trajectory = data[:, joint_idx, coord_idx]
                # Only filter if we have valid (non-NaN) data
                if not np.all(np.isnan(trajectory)):
                    # Find valid indices
                    valid_mask = ~np.isnan(trajectory)
                    if np.sum(valid_mask) >= min_frames_required:  # Need enough points to filter
                        # Filter only the valid portion
                        filtered_data[:, joint_idx, coord_idx] = filtfilt(b, a, trajectory)
    elif data.ndim == 2:  # (frames, features)
        for feature_idx in range(data.shape[1]):
            trajectory = data[:, feature_idx]
            if not np.all(np.isnan(trajectory)):
                valid_mask = ~np.isnan(trajectory)
                if np.sum(valid_mask) >= min_frames_required:
                    filtered_data[:, feature_idx] = filtfilt(b, a, trajectory)
    else:
        # For 1D, just apply directly
        if not np.all(np.isnan(data)):
            filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def interpolate_missing_frames(data: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN/missing frames using linear interpolation.
    
    Args:
        data: Input pose data with shape (frames, joints, 3)
    
    Returns:
        Data with NaN values interpolated
    
    Raises:
        ValueError: If all frames are NaN
    """
    if np.all(np.isnan(data)):
        raise ValueError("Cannot interpolate: all frames are NaN")
    
    interpolated_data = data.copy()
    
    if data.ndim == 3:  # (frames, joints, coords)
        frames, joints, coords = data.shape
        
        for joint_idx in range(joints):
            for coord_idx in range(coords):
                trajectory = data[:, joint_idx, coord_idx]
                
                # Find valid (non-NaN) indices
                valid_mask = ~np.isnan(trajectory)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    # All NaN for this trajectory, skip
                    continue
                elif len(valid_indices) == 1:
                    # Only one valid point, fill all with that value
                    interpolated_data[:, joint_idx, coord_idx] = trajectory[valid_indices[0]]
                else:
                    # Interpolate
                    interp_func = interp1d(
                        valid_indices,
                        trajectory[valid_indices],
                        kind='linear',
                        bounds_error=False,
                        fill_value=(trajectory[valid_indices[0]], trajectory[valid_indices[-1]])
                    )
                    all_indices = np.arange(frames)
                    interpolated_data[:, joint_idx, coord_idx] = interp_func(all_indices)
    
    elif data.ndim == 2:  # (frames, features)
        frames, features = data.shape
        
        for feature_idx in range(features):
            trajectory = data[:, feature_idx]
            
            valid_mask = ~np.isnan(trajectory)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue
            elif len(valid_indices) == 1:
                interpolated_data[:, feature_idx] = trajectory[valid_indices[0]]
            else:
                interp_func = interp1d(
                    valid_indices,
                    trajectory[valid_indices],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(trajectory[valid_indices[0]], trajectory[valid_indices[-1]])
                )
                all_indices = np.arange(frames)
                interpolated_data[:, feature_idx] = interp_func(all_indices)
    
    return interpolated_data


def enforce_bone_lengths(
    pose_3d: np.ndarray,
    constraints: Dict[Tuple[int, int], Tuple[float, float]] = None
) -> np.ndarray:
    """
    Enforce anatomical bone-length constraints on 3D pose data.
    
    Clamps joint-pair distances to anatomically plausible ranges to correct
    artifacts from refraction, occlusion, or lifting errors.
    
    Args:
        pose_3d: Input pose data with shape (frames, joints, 3)
        constraints: Dictionary mapping (joint1_idx, joint2_idx) to (min_length, max_length).
                    If None, uses default BONE_LENGTH_CONSTRAINTS.
    
    Returns:
        Pose data with bone lengths enforced
    """
    if constraints is None:
        constraints = BONE_LENGTH_CONSTRAINTS
    
    corrected_pose = pose_3d.copy()
    
    if pose_3d.ndim != 3:
        # Only works on 3D pose data
        return corrected_pose
    
    frames, joints, coords = pose_3d.shape
    
    # Apply constraints for each bone
    for (joint1_idx, joint2_idx), (min_length, max_length) in constraints.items():
        # Skip if joint indices are out of bounds
        if joint1_idx >= joints or joint2_idx >= joints:
            continue
        
        for frame_idx in range(frames):
            joint1_pos = corrected_pose[frame_idx, joint1_idx, :]
            joint2_pos = corrected_pose[frame_idx, joint2_idx, :]
            
            # Skip if either joint has NaN values
            if np.any(np.isnan(joint1_pos)) or np.any(np.isnan(joint2_pos)):
                continue
            
            # Compute current bone vector and length
            bone_vector = joint2_pos - joint1_pos
            current_length = np.linalg.norm(bone_vector)
            
            if current_length < 1e-6:
                # Degenerate case: joints are at same position
                continue
            
            # Clamp to constraints
            if current_length < min_length:
                # Stretch to minimum length
                scale_factor = min_length / current_length
                corrected_pose[frame_idx, joint2_idx, :] = joint1_pos + bone_vector * scale_factor
            elif current_length > max_length:
                # Shrink to maximum length
                scale_factor = max_length / current_length
                corrected_pose[frame_idx, joint2_idx, :] = joint1_pos + bone_vector * scale_factor
    
    return corrected_pose


def post_process(
    data: np.ndarray,
    cutoff: float = 3.0,
    fs: float = 30.0,
    order: int = 2,
    interpolate: bool = True,
    enforce_constraints: bool = True
) -> np.ndarray:
    """
    Complete post-processing pipeline for 3D pose data.
    
    Applies the following steps in order:
    1. Interpolate NaN/missing frames (if enabled)
    2. Apply Butterworth low-pass filter for smoothing
    3. Enforce anatomical bone-length constraints (if enabled)
    
    Args:
        data: Input pose data with shape (frames, joints, 3)
        cutoff: Low-pass filter cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Butterworth filter order
        interpolate: Whether to interpolate missing frames
        enforce_constraints: Whether to enforce bone-length constraints
    
    Returns:
        Post-processed pose data with same shape as input
    
    Raises:
        ValueError: If input shape is invalid or all data is NaN
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D pose array with shape (frames, joints, 3), got shape {data.shape}")
    
    result = data.copy()
    
    # Step 1: Interpolate missing frames
    if interpolate:
        result = interpolate_missing_frames(result)
    
    # Step 2: Apply smoothing filter
    result = apply_butterworth_filter(result, cutoff=cutoff, fs=fs, order=order)
    
    # Step 3: Enforce bone-length constraints
    if enforce_constraints:
        result = enforce_bone_lengths(result)
    
    return result
