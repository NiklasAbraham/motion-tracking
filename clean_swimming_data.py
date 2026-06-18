import argparse
from pathlib import Path

import numpy as np

from src.post_processor import post_process


def _load_pose_array(input_path: Path) -> np.ndarray:
    """Load 3D pose array from .npy or .npz file."""
    if input_path.suffix == ".npz":
        loaded = np.load(input_path)
        if "poses" not in loaded:
            raise ValueError("NPZ input must contain a 'poses' array")
        return loaded["poses"]
    return np.load(input_path)


def clean_swimming_data(input_path: Path, output_path: Path, cutoff: float = 3, fs: float = 30, order: int = 2) -> None:
    """
    Apply aquatic post-processing to 3D swimming pose data.
    
    Args:
        input_path: Path to input .npy/.npz file with 3D pose data
        output_path: Path to output .npy file
        cutoff: Low-pass filter cutoff frequency in Hz
        fs: Sampling rate in Hz
        order: Butterworth filter order
    """
    raw_3d_data = _load_pose_array(input_path)
    if raw_3d_data.ndim != 3:
        raise ValueError("Expected 3D pose array shaped [frames, joints, coords]")

    print("Applying aquatic noise filters...")
    corrected_data = post_process(raw_3d_data, cutoff=cutoff, fs=fs, order=order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, corrected_data)
    print(f"Post-processing complete. Saved cleaned output to '{output_path}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply aquatic post-processing to 3D swimming pose data.")
    parser.add_argument("input", type=Path, help="Path to DiffPose output .npy/.npz file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("swimmer_3d_cleaned.npy"),
        help="Path to output cleaned .npy file (default: swimmer_3d_cleaned.npy)",
    )
    parser.add_argument("--cutoff", type=float, default=3.0, help="Low-pass cutoff frequency in Hz")
    parser.add_argument("--fs", type=float, default=30.0, help="Sampling rate in Hz")
    parser.add_argument("--order", type=int, default=2, help="Butterworth filter order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_swimming_data(args.input, args.output, cutoff=args.cutoff, fs=args.fs, order=args.order)


if __name__ == "__main__":
    main()
