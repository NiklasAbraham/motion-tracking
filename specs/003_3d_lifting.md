# 003 – 3D Pose Lifting

## Purpose

Convert 2D keypoint sequences into temporally consistent 3D joint trajectories.

## Requirements

1. Accept 2D keypoint data (NPZ format from 2D pose estimation stage).
2. Map YOLO 17-joint format to DiffPose 16-joint format.
3. Produce 3D coordinates for each joint per frame.
4. Maintain temporal coherence across consecutive frames.
5. Output as NumPy array of shape `(frames, joints, 3)`.

## Model

- **DiffPose** – diffusion-based 2D-to-3D lifting.
- Pretrained on Human3.6M dataset.

## Interface

```python
def lift_2d_to_3d(input_2d_path: Path, output_3d_path: Path, config: LiftingConfig) -> int:
    """Return number of frames lifted."""
```

## Keypoint Mapping

| YOLO Index | Joint Name | DiffPose Index |
|-----------|-----------|----------------|
| 0 | Nose | 9 (Head) |
| 5 | Left Shoulder | 11 |
| 6 | Right Shoulder | 14 |
| 7 | Left Elbow | 12 |
| 8 | Right Elbow | 15 |
| 9 | Left Wrist | 13 |
| 10 | Right Wrist | 16 |
| 11 | Left Hip | 4 |
| 12 | Right Hip | 1 |
| 13 | Left Knee | 5 |
| 14 | Right Knee | 2 |
| 15 | Left Ankle | 6 |
| 16 | Right Ankle | 3 |

## Output Format

NPY file with array shape `(frames, 16, 3)` containing `[x, y, z]` coordinates in meters.
