# 001 – Input Handling

## Purpose

Define how the system accepts and validates video input before processing.

## Requirements

1. Accept MP4 video files as the primary input format.
2. Validate that the file exists, is readable, and contains at least one video frame.
3. Extract video metadata (resolution, FPS, duration, codec).
4. Support videos of arbitrary resolution (internally resize if needed).
5. Reject corrupted or zero-length files with clear error messages.

## Input Format

| Field | Constraint |
|-------|-----------|
| Container | MP4 (H.264 / H.265 codec) |
| Min Resolution | 480×360 |
| Max Resolution | 4K (3840×2160) |
| Min Duration | 1 second |
| FPS | 24–120 fps |

## Interface

```python
def validate_video(path: Path) -> VideoMetadata:
    """Return metadata or raise InvalidVideoError."""
```

## Output

A `VideoMetadata` dataclass containing:
- `width: int`
- `height: int`
- `fps: float`
- `frame_count: int`
- `duration_seconds: float`
- `codec: str`
