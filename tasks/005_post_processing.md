# Task 005 – Post-Processing & Smoothing

## Summary

Build the module that applies temporal smoothing, bone-length constraints, and refraction correction to raw 3D pose data.

## Spec Reference

`specs/004_post_processing.md`

## Deliverables

- `src/post_processor.py` – smoothing and correction logic
- `tests/test_post_processor.py` – unit tests

## Acceptance Criteria

1. Applies Butterworth low-pass filter to remove high-frequency noise.
2. Enforces anatomical bone-length constraints.
3. Interpolates NaN/missing frames.
4. Accepts configurable filter parameters (cutoff, fs, order).
5. Output shape matches input shape.

## Dependencies (External)

- `scipy` (signal processing)
- `numpy`

## Dependencies (Internal)

- None – pure data processing module.

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T5.1 | Smoothing reduces noise | Noisy sine wave + random jitter | Output closer to clean sine |
| T5.2 | Bone-length enforcement | Joint pair exceeding max length | Clamped to anatomical max |
| T5.3 | NaN interpolation | 3 consecutive NaN frames | Interpolated values filled |
| T5.4 | Shape preservation | (200, 16, 3) input | (200, 16, 3) output |
| T5.5 | Custom filter params | cutoff=5, fs=60, order=4 | Filter applied with those params |
| T5.6 | All-NaN input | Entire array is NaN | Raises ValueError or returns NaN |
| T5.7 | Single frame input | (1, 16, 3) array | Returns unchanged (no filtering possible) |

## Parallelism

✅ Can run independently. Self-contained numerical processing with no external service calls.
