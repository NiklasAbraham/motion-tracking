import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def extract_2d_keypoints(video_path: Path, output_path: Path, model_name: str = "yolov8x-pose.pt") -> int:
    """Extract 2D keypoints from a swimming video and save them to NPZ."""
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    all_frames_2d = []

    print("Starting 2D Keypoint Extraction... (Processing offline)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            keypoints = r.keypoints.data.cpu().numpy()
            if len(keypoints) > 0:
                all_frames_2d.append(keypoints[0])

    cap.release()

    poses = np.array(all_frames_2d)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, poses=poses)
    print(f"Extraction complete! Saved {len(poses)} frames to '{output_path}'")

    return len(poses)


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
