# 002 – 2D Pose Estimation

## Purpose

Extract per-frame 2D keypoints from the input swimming video.

## Requirements

1. Process every frame of the input video.
2. Detect the primary swimmer when multiple people are present.
3. Output 17-joint COCO keypoints with confidence scores per frame.
4. Handle frames with no detection (fill with NaN markers).
5. Store results in a NumPy-compatible format (NPZ).

## Model

- **YOLOv8x-pose** via the Ultralytics library.
- Offline batch processing (no streaming requirement).

## Interface

```python
def extract_2d_keypoints(video_path: Path, output_path: Path, model_name: str) -> int:
    """Return the number of frames processed."""
```

## Output Format

NPZ file with key `poses` → NumPy array of shape `(frames, 17, 3)` where the last dimension is `[x, y, confidence]`.

## Edge Cases

- No detection in frame → row filled with `np.nan`.
- Multiple detections → select highest average confidence.
- Very low confidence frame → keep data, flag for downstream filtering.
