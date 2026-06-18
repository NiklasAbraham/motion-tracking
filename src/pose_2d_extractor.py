"""2D pose estimation module for extracting keypoints from video frames.

This module wraps YOLOv8x-pose for offline batch processing of swimming videos.
It extracts 17 COCO-format keypoints per frame with confidence scores.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


def _select_primary_detection(keypoints: np.ndarray) -> np.ndarray:
    """Select the detection with the highest average keypoint confidence.
    
    When multiple people are detected in a frame, this function selects
    the primary subject based on the highest average confidence across
    all keypoints.
    
    Args:
        keypoints: Array of shape (num_detections, 17, 3) where last dimension
                  is [x, y, confidence].
    
    Returns:
        Array of shape (17, 3) representing the selected detection.
    """
    if len(keypoints) == 1:
        return keypoints[0]
    confidence_means = keypoints[:, :, 2].mean(axis=1)
    return keypoints[int(np.argmax(confidence_means))]


def extract_2d_keypoints(
    video_path: Path,
    output_path: Path,
    model_name: str = "yolov8x-pose.pt",
    verbose: bool = True
) -> int:
    """Extract 2D keypoints from a video and save to NPZ format.
    
    Processes every frame of the input video, detects poses using YOLOv8x-pose,
    and stores the results as a NumPy array. When multiple people are detected,
    selects the one with highest average confidence. Frames with no detection
    are filled with NaN values.
    
    Args:
        video_path: Path to the input video file.
        output_path: Path where the output NPZ file will be saved.
        model_name: Name of the YOLO pose model checkpoint (default: yolov8x-pose.pt).
        verbose: Whether to print progress messages (default: True).
    
    Returns:
        Number of frames processed.
    
    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If video cannot be opened.
    
    Output Format:
        NPZ file with key 'poses' containing array of shape (frames, 17, 3)
        where last dimension is [x, y, confidence]. COCO format with 17 joints.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    all_frames_2d = []

    if verbose:
        print("Starting 2D Keypoint Extraction... (Processing offline)")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # Default: fill with NaN if no detection
        frame_pose = np.full((17, 3), np.nan, dtype=float)
        
        for r in results:
            keypoints = r.keypoints.data.cpu().numpy()
            if len(keypoints) > 0:
                frame_pose = _select_primary_detection(keypoints)
                break

        all_frames_2d.append(frame_pose)
        frame_count += 1

    cap.release()

    poses = np.array(all_frames_2d)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as NPZ with 'poses' key
    np.savez(output_path, poses=poses)
    
    if verbose:
        print(f"Extraction complete! Saved {len(poses)} frames to '{output_path}'")

    return len(poses)


def load_poses(npz_path: Path) -> np.ndarray:
    """Load poses from an NPZ file.
    
    Args:
        npz_path: Path to the NPZ file containing poses.
    
    Returns:
        Array of shape (frames, 17, 3) with [x, y, confidence] per keypoint.
    
    Raises:
        FileNotFoundError: If npz_path does not exist.
        KeyError: If 'poses' key is not found in the NPZ file.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    if 'poses' not in data:
        raise KeyError("NPZ file does not contain 'poses' key")
    
    return data['poses']
