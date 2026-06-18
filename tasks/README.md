# Tasks

This folder contains the development tasks for building the motion-tracking pipeline. Each task is **independent** and can be developed and tested in parallel.

## Task List

| Task | Module | Can Run in Parallel |
|------|--------|-------------------|
| `001_input_validator.md` | Video file validation | ✅ |
| `002_2d_pose_estimation.md` | 2D keypoint extraction | ✅ |
| `003_keypoint_mapper.md` | YOLO → DiffPose format conversion | ✅ |
| `004_3d_lifting.md` | DiffPose 3D lifting integration | ✅ |
| `005_post_processing.md` | Smoothing & bone constraints | ✅ |
| `006_3d_visualization.md` | 3D skeleton viewer | ✅ |
| `007_pipeline_orchestrator.md` | End-to-end CLI orchestration | ✅ |

## Independence Guarantee

Each task:
- Has its own input/output contract defined in the specs.
- Can be tested with mocked dependencies.
- Has no shared mutable state with other tasks.
- Produces a standalone module in `src/`.

## Execution Order (Runtime)

While tasks can be **developed** in parallel, at runtime the pipeline executes sequentially:

```
Task 001 → Task 002 → Task 003 → Task 004 → Task 005 → Task 006
                              (Task 007 orchestrates all)
```

## Test Strategy

Each task includes a test case table. Tests use:
- **Unit tests** with mocked I/O and model inference.
- **Synthetic data** (NumPy arrays) for numerical validation.
- **pytest** as the test runner.
