# Task 002 – 2D Pose Estimation Module

## Summary

Build the module that processes video frames and extracts 2D keypoints using YOLOv8x-pose. This wraps the existing `extract_2d_joints.py` into a well-tested, modular component.

## Spec Reference

`specs/002_2d_pose_estimation.md`

## Deliverables

- `src/pose_2d_extractor.py` – extraction logic
- `tests/test_pose_2d_extractor.py` – unit tests (with mocked model)

## Acceptance Criteria

1. Processes all frames and outputs shape `(N, 17, 3)` array.
2. Selects highest-confidence detection when multiple people present.
3. Fills NaN for frames with no detection.
4. Saves output as NPZ with key `poses`.
5. Reports progress (frame count processed).

## Dependencies (External)

- `ultralytics` (YOLO)
- `opencv-python` (cv2)
- `numpy`

## Dependencies (Internal)

- None – receives a video path; does not depend on other task outputs at dev time.

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T2.1 | Single person, clear frame | Mock frame with 1 detection | Correct 17×3 keypoints |
| T2.2 | Multiple people | Mock frame with 3 detections | Selects highest confidence |
| T2.3 | No detection in frame | Mock frame with 0 detections | Row of NaN values |
| T2.4 | Full video processing | 10-frame mock video | Output shape (10, 17, 3) |
| T2.5 | Output file format | Any valid input | NPZ file with 'poses' key |
| T2.6 | Empty video (0 frames) | Empty video capture | Returns 0, saves empty array |
| T2.7 | Low confidence detection | Mock frame, all conf < 0.1 | Still stored (not filtered) |

## Parallelism

✅ Can run independently. Uses only a video file path as input. Model inference is self-contained.
