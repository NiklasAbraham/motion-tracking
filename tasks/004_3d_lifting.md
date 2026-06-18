# Task 004 – 3D Pose Lifting Integration

## Summary

Build the module that invokes DiffPose to lift mapped 2D keypoints into 3D joint coordinates.

## Spec Reference

`specs/003_3d_lifting.md`

## Deliverables

- `src/pose_3d_lifter.py` – lifting orchestration logic
- `tests/test_pose_3d_lifter.py` – unit tests (with mocked DiffPose)

## Acceptance Criteria

1. Accepts mapped 2D keypoints in DiffPose format `(frames, 16, 2)`.
2. Produces 3D output of shape `(frames, 16, 3)`.
3. Handles DiffPose checkpoint loading and configuration.
4. Reports errors clearly if model files are missing.
5. Saves output as NPY file.

## Dependencies (External)

- `torch` (PyTorch)
- `numpy`
- DiffPose repository (cloned separately)

## Dependencies (Internal)

- None at development time. At runtime, consumes output of Task 003.

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T4.1 | Mocked lifting | Synthetic (50, 16, 2) input | Output shape (50, 16, 3) |
| T4.2 | Missing checkpoint | Invalid checkpoint path | Raises FileNotFoundError with message |
| T4.3 | Invalid config | Bad YAML config path | Raises ConfigError |
| T4.4 | Output file creation | Valid input | NPY file written to disk |
| T4.5 | Zero-frame input | (0, 16, 2) array | Returns (0, 16, 3) array |
| T4.6 | Temporal consistency | Smooth 2D input | 3D output has no sudden jumps (< threshold) |
| T4.7 | NaN handling | Frames with NaN | NaN frames remain NaN in output |

## Parallelism

✅ Can run independently. Model loading and inference are self-contained. Tests use mocks.
