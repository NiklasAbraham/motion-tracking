# 004 – Post-Processing

## Purpose

Smooth 3D trajectories and correct aquatic-specific artifacts (refraction, splash noise).

## Requirements

1. Apply temporal smoothing (Butterworth low-pass filter).
2. Enforce anatomical bone-length constraints.
3. Apply refraction correction for underwater footage.
4. Handle NaN/missing frames via interpolation.
5. Output cleaned 3D pose data.

## Interface

```python
def clean_swimming_data(input_path: Path, output_path: Path, cutoff: float, fs: float, order: int) -> None:
    """Smooth and correct 3D pose data in-place."""
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| cutoff | 3.0 Hz | Low-pass filter cutoff frequency |
| fs | 30.0 Hz | Video sampling rate |
| order | 2 | Butterworth filter order |

## Output Format

NPY file with array shape `(frames, joints, 3)` — same structure as input but smoothed.
