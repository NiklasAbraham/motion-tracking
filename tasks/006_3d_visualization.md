# Task 006 – 3D Visualization Renderer

## Summary

Build the module that takes cleaned 3D pose data and produces an interactive 3D visualization (HTML viewer, matplotlib animation, or MP4 export).

## Spec Reference

`specs/005_3d_visualization.md`

## Deliverables

- `src/visualizer.py` – visualization rendering logic
- `tests/test_visualizer.py` – unit tests

## Acceptance Criteria

1. Loads 3D pose data from NPY file.
2. Renders stick-figure skeleton with correct bone connections.
3. Supports at least one output format (HTML with Three.js or matplotlib).
4. Includes playback controls (play, pause, frame scrub).
5. Allows camera orbit controls (rotate, zoom, pan).
6. Displays frame counter / timestamp overlay.

## Dependencies (External)

- `matplotlib` (for matplotlib mode)
- `numpy`
- `jinja2` (optional, for HTML template rendering)

## Dependencies (Internal)

- None – consumes an NPY file independently.

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T6.1 | Load valid NPY | (100, 16, 3) pose file | Data loaded without error |
| T6.2 | Skeleton bone count | Standard 16-joint input | Correct number of bones rendered |
| T6.3 | HTML output | Valid pose data, format="html" | Generates valid HTML file |
| T6.4 | Matplotlib output | Valid pose data, format="matplotlib" | Generates animation object |
| T6.5 | Frame count display | 100-frame input | Frame counter shows 0–99 |
| T6.6 | Empty pose data | (0, 16, 3) array | Raises ValueError or shows empty scene |
| T6.7 | Single frame | (1, 16, 3) array | Renders static pose |

## Parallelism

✅ Can run independently. Only requires a pose data file as input — no coupling to other modules.
