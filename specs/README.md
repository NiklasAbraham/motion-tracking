# Specs

This folder contains the technical specifications for the motion-tracking pipeline.

## Files

| File | Topic |
|------|-------|
| `000_general.md` | Overall project goals, pipeline overview, and technology stack |
| `001_input_handling.md` | Video input validation and metadata extraction |
| `002_2d_pose_estimation.md` | Per-frame 2D keypoint extraction with YOLOv8x-pose |
| `003_3d_lifting.md` | 2D-to-3D pose lifting with DiffPose |
| `004_post_processing.md` | Temporal smoothing and aquatic corrections |
| `005_3d_visualization.md` | Interactive 3D skeletal model viewer |
| `006_pipeline_orchestration.md` | End-to-end pipeline orchestration |
