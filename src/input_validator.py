"""Input video validation module.

Validates MP4 files and extracts metadata before processing begins.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2


class InvalidVideoError(Exception):
    """Raised when a video file is invalid, corrupted, or does not meet requirements."""


@dataclass
class VideoMetadata:
    """Metadata extracted from a valid video file."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str


_MIN_WIDTH = 480
_MIN_HEIGHT = 360


def _decode_fourcc(fourcc_int: int) -> str:
    """Decode a FourCC integer into a human-readable codec string."""
    return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))


def validate_video(path: Path) -> VideoMetadata:
    """Validate a video file and return its metadata.

    Parameters
    ----------
    path : Path
        Path to the video file to validate.

    Returns
    -------
    VideoMetadata
        Extracted metadata for the valid video.

    Raises
    ------
    InvalidVideoError
        If the file does not exist, is empty, cannot be opened as video,
        or does not meet minimum resolution requirements.
    """
    path = Path(path)

    if not path.exists():
        raise InvalidVideoError(f"File does not exist: {path}")

    if path.stat().st_size == 0:
        raise InvalidVideoError(f"File is empty (0 bytes): {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise InvalidVideoError(f"Cannot open file as video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Verify we can actually read at least one frame
        ret, _ = cap.read()
        if not ret:
            raise InvalidVideoError(f"Cannot read any frames from video: {path}")

        if width < _MIN_WIDTH or height < _MIN_HEIGHT:
            raise InvalidVideoError(
                f"Resolution {width}x{height} is below minimum "
                f"{_MIN_WIDTH}x{_MIN_HEIGHT}: {path}"
            )

        codec = _decode_fourcc(fourcc_int)
        duration_seconds = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
            codec=codec,
        )
    finally:
        cap.release()
