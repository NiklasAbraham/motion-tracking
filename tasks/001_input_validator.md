# Task 001 – Input Video Validator

## Summary

Build a standalone module that validates an MP4 file and extracts its metadata before any processing begins.

## Spec Reference

`specs/001_input_handling.md`

## Deliverables

- `src/input_validator.py` – validation logic
- `tests/test_input_validator.py` – unit tests

## Acceptance Criteria

1. Returns a `VideoMetadata` dataclass for valid MP4 files.
2. Raises `InvalidVideoError` for missing, corrupted, or empty files.
3. Correctly extracts width, height, fps, frame_count, duration, and codec.
4. Rejects files below minimum resolution (480×360).
5. Works independently with no dependency on other pipeline stages.

## Dependencies (External)

- `opencv-python` (cv2)
- Python standard library (`pathlib`, `dataclasses`)

## Dependencies (Internal)

- None – fully independent.

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T1.1 | Valid MP4 file | sample_valid.mp4 | Returns VideoMetadata with correct fields |
| T1.2 | File does not exist | /nonexistent/path.mp4 | Raises InvalidVideoError |
| T1.3 | Empty file (0 bytes) | empty.mp4 | Raises InvalidVideoError |
| T1.4 | Non-video file renamed to .mp4 | text.mp4 | Raises InvalidVideoError |
| T1.5 | Resolution below minimum | tiny_100x100.mp4 | Raises InvalidVideoError |
| T1.6 | 4K video file | sample_4k.mp4 | Returns VideoMetadata with 3840×2160 |
| T1.7 | Variable FPS video | vfr_video.mp4 | Returns average FPS in metadata |

## Parallelism

✅ Can run independently. No shared state or dependencies on other tasks.
