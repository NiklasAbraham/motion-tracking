# 005 – 3D Visualization

## Purpose

Render the cleaned 3D skeletal data as an interactive 3D model that users can view and rotate.

## Requirements

1. Load cleaned 3D pose data (NPY format).
2. Render a stick-figure skeleton with joint markers.
3. Support playback (play, pause, scrub timeline).
4. Allow camera orbit (rotate, zoom, pan).
5. Display frame counter and timestamp.
6. Export options: animated GIF, MP4 recording, or interactive HTML.

## Interface

```python
def visualize_3d(pose_data_path: Path, output_path: Path, format: str = "html") -> None:
    """Generate an interactive 3D visualization."""
```

## Visualization Modes

| Mode | Description |
|------|-------------|
| Interactive HTML | Three.js-based viewer, self-contained HTML file |
| Matplotlib | Rotating 3D matplotlib animation |
| MP4 Export | Pre-rendered video of the 3D skeleton |

## Skeleton Definition

Bones are defined as joint-pair connections:
- Head → Neck
- Neck → Left Shoulder, Right Shoulder
- Left Shoulder → Left Elbow → Left Wrist
- Right Shoulder → Right Elbow → Right Wrist
- Neck → Hip Center
- Hip Center → Left Hip → Left Knee → Left Ankle
- Hip Center → Right Hip → Right Knee → Right Ankle
