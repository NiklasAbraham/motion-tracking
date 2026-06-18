"""Keypoint Format Mapper

This module converts YOLO 17-joint COCO keypoints to DiffPose 16-joint Human3.6M format.

YOLO format: (frames, 17, 3) - 17 joints with [x, y, confidence]
DiffPose format: (frames, 16, 2) - 16 joints with [x, y] only

Mapping from YOLO Index to DiffPose Index (using 0-based array indexing):
    YOLO 0 (Nose) -> DiffPose 8 (Head; spec lists as 9)
    YOLO 5 (Left Shoulder) -> DiffPose 10 (spec lists as 11)
    YOLO 6 (Right Shoulder) -> DiffPose 13 (spec lists as 14)
    YOLO 7 (Left Elbow) -> DiffPose 11 (spec lists as 12)
    YOLO 8 (Right Elbow) -> DiffPose 14 (spec lists as 15)
    YOLO 9 (Left Wrist) -> DiffPose 12 (spec lists as 13)
    YOLO 10 (Right Wrist) -> DiffPose 15 (spec lists as 16)
    YOLO 11 (Left Hip) -> DiffPose 3 (spec lists as 4)
    YOLO 12 (Right Hip) -> DiffPose 0 (spec lists as 1)
    YOLO 13 (Left Knee) -> DiffPose 4 (spec lists as 5)
    YOLO 14 (Right Knee) -> DiffPose 1 (spec lists as 2)
    YOLO 15 (Left Ankle) -> DiffPose 5 (spec lists as 6)
    YOLO 16 (Right Ankle) -> DiffPose 2 (spec lists as 3)
"""

import numpy as np
from typing import Union


# Mapping from YOLO joint index to DiffPose joint index
# DiffPose uses 0-based indexing, but the spec shows 1-based indexing
# We need to convert: spec index - 1 = array index
YOLO_TO_DIFFPOSE_MAPPING = {
    0: 8,   # Nose -> Head (spec says 9, but array index is 8)
    5: 10,  # Left Shoulder -> 11 in spec, 10 in array
    6: 13,  # Right Shoulder -> 14 in spec, 13 in array
    7: 11,  # Left Elbow -> 12 in spec, 11 in array
    8: 14,  # Right Elbow -> 15 in spec, 14 in array
    9: 12,  # Left Wrist -> 13 in spec, 12 in array
    10: 15, # Right Wrist -> 16 in spec, 15 in array
    11: 3,  # Left Hip -> 4 in spec, 3 in array
    12: 0,  # Right Hip -> 1 in spec, 0 in array
    13: 4,  # Left Knee -> 5 in spec, 4 in array
    14: 1,  # Right Knee -> 2 in spec, 1 in array
    15: 5,  # Left Ankle -> 6 in spec, 5 in array
    16: 2,  # Right Ankle -> 3 in spec, 2 in array
}


def map_yolo_to_diffpose(yolo_keypoints: np.ndarray) -> np.ndarray:
    """Convert YOLO 17-joint keypoints to DiffPose 16-joint format.
    
    Args:
        yolo_keypoints: Array of shape (frames, 17, 3) containing YOLO keypoints
                       with [x, y, confidence] for each joint.
    
    Returns:
        Array of shape (frames, 16, 2) containing DiffPose keypoints with [x, y]
        for each joint.
    
    Raises:
        ValueError: If input shape is not (frames, 17, 3) or (frames, 17, 2).
    
    Examples:
        >>> yolo_kpts = np.random.rand(100, 17, 3)
        >>> diffpose_kpts = map_yolo_to_diffpose(yolo_kpts)
        >>> diffpose_kpts.shape
        (100, 16, 2)
    """
    # Validate input shape
    if yolo_keypoints.ndim != 3:
        raise ValueError(
            f"Expected 3D array with shape (frames, 17, 3), "
            f"got {yolo_keypoints.ndim}D array with shape {yolo_keypoints.shape}"
        )
    
    frames, num_joints, coords = yolo_keypoints.shape
    
    if num_joints != 17:
        raise ValueError(
            f"Expected 17 joints in YOLO format, got {num_joints} joints"
        )
    
    if coords not in [2, 3]:
        raise ValueError(
            f"Expected 2 or 3 coordinate dimensions, got {coords}"
        )
    
    # Initialize output array with NaN
    diffpose_keypoints = np.full((frames, 16, 2), np.nan, dtype=yolo_keypoints.dtype)
    
    # Extract only x, y coordinates (drop confidence if present)
    yolo_xy = yolo_keypoints[:, :, :2]
    
    # Map each YOLO joint to its corresponding DiffPose joint
    for yolo_idx, diffpose_idx in YOLO_TO_DIFFPOSE_MAPPING.items():
        diffpose_keypoints[:, diffpose_idx, :] = yolo_xy[:, yolo_idx, :]
    
    return diffpose_keypoints
