"""CLI wrapper for the 2D pose extraction module.

This script provides a command-line interface for extracting 2D keypoints
from swimming videos using the src.pose_2d_extractor module.
"""

import argparse
from pathlib import Path

from src.pose_2d_extractor import extract_2d_keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 2D swimmer joints using YOLOv8x-pose.")
    parser.add_argument("video", type=Path, help="Path to the input swimming video file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("swimmer_2d_output.npz"),
        help="Path to output NPZ file (default: swimmer_2d_output.npz)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x-pose.pt",
        help="Ultralytics pose model checkpoint (default: yolov8x-pose.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_2d_keypoints(args.video, args.output, model_name=args.model)


if __name__ == "__main__":
    main()
