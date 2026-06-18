# Task 003 – Keypoint Format Mapper

## Summary

Build a utility module that converts YOLO 17-joint COCO keypoints to the DiffPose 16-joint Human3.6M format required for 3D lifting.

## Spec Reference

`specs/003_3d_lifting.md` (Keypoint Mapping table)

## Deliverables

- `src/keypoint_mapper.py` – mapping logic
- `tests/test_keypoint_mapper.py` – unit tests

## Acceptance Criteria

1. Accepts a `(frames, 17, 3)` array (YOLO format).
2. Outputs a `(frames, 16, 2)` array (DiffPose format, xy only).
3. Correctly maps each YOLO joint index to its DiffPose equivalent.
4. Handles NaN frames gracefully (passes NaN through).
5. Drops the nose joint or maps it to head as defined in spec.

## Dependencies (External)

- `numpy`

## Dependencies (Internal)

- None – pure data transformation with no I/O dependencies.

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T3.1 | Basic mapping | Known (1, 17, 3) array | Correct (1, 16, 2) output |
| T3.2 | Multi-frame mapping | (100, 17, 3) array | Output shape (100, 16, 2) |
| T3.3 | NaN frame passthrough | Frame of all NaN | Output frame is all NaN |
| T3.4 | Joint index correctness | Left shoulder at YOLO[5] | Maps to DiffPose[11] |
| T3.5 | Confidence stripping | Input has confidence dim | Output has only x, y |
| T3.6 | Zero-frame input | (0, 17, 3) array | Returns (0, 16, 2) array |
| T3.7 | Invalid input shape | (10, 15, 3) array | Raises ValueError |

## Parallelism

✅ Can run independently. Pure NumPy transformation, no external dependencies.
