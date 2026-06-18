# 000 – General Overview

## Project Goal

Build an offline pipeline that accepts a single-camera swimming video (MP4) and produces an interactive 3D skeletal model visualization of the swimmer.

## High-Level Pipeline

```
Input Video (MP4)
      │
      ▼
┌─────────────────────┐
│ 2D Pose Estimation  │  (per-frame keypoint extraction)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 2D → 3D Lifting     │  (temporal diffusion model)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Post-Processing      │  (smoothing, bone constraints, refraction)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 3D Visualization     │  (interactive 3D model viewer)
└─────────────────────┘
```

## Constraints

- **Offline processing** – no real-time requirements.
- **Single camera** – monocular input only.
- **Domain** – swimming / underwater footage.
- **Output** – 3D skeletal animation viewable in a browser or desktop viewer.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| 2D Pose | YOLOv8x-pose (Ultralytics) |
| 3D Lifting | DiffPose (temporal diffusion) |
| Post-Processing | SciPy (Butterworth filter), NumPy |
| Visualization | Three.js or Matplotlib 3D animation |
| Language | Python 3.10+ |

## Quality Attributes

- Accuracy: sub-5 cm joint error on visible joints.
- Robustness: handle splash occlusion, bubbles, and refraction artifacts.
- Modularity: each pipeline stage is an independent, testable module.
