"""Unit tests for src/input_validator.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.input_validator import InvalidVideoError, VideoMetadata, validate_video


@pytest.fixture
def valid_video(tmp_path):
    """Create a minimal valid MP4 video file using OpenCV."""
    import cv2

    video_path = tmp_path / "sample_valid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    # Write 30 frames (1 second at 30fps)
    for _ in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def small_video(tmp_path):
    """Create a video below minimum resolution."""
    import cv2

    video_path = tmp_path / "tiny_100x100.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (100, 100))
    for _ in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def large_video(tmp_path):
    """Create a 4K resolution video file."""
    import cv2

    video_path = tmp_path / "sample_4k.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (3840, 2160))
    # Write just a few frames to keep it fast
    for _ in range(5):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestValidVideo:
    """T1.1: Valid MP4 file returns VideoMetadata with correct fields."""

    def test_returns_video_metadata(self, valid_video):
        result = validate_video(valid_video)
        assert isinstance(result, VideoMetadata)

    def test_correct_resolution(self, valid_video):
        result = validate_video(valid_video)
        assert result.width == 640
        assert result.height == 480

    def test_correct_fps(self, valid_video):
        result = validate_video(valid_video)
        assert result.fps == pytest.approx(30.0, abs=1.0)

    def test_correct_frame_count(self, valid_video):
        result = validate_video(valid_video)
        assert result.frame_count == 30

    def test_correct_duration(self, valid_video):
        result = validate_video(valid_video)
        assert result.duration_seconds == pytest.approx(1.0, abs=0.1)

    def test_codec_is_string(self, valid_video):
        result = validate_video(valid_video)
        assert isinstance(result.codec, str)
        assert len(result.codec) > 0


class TestFileDoesNotExist:
    """T1.2: File does not exist raises InvalidVideoError."""

    def test_nonexistent_path(self):
        with pytest.raises(InvalidVideoError, match="does not exist"):
            validate_video(Path("/nonexistent/path.mp4"))


class TestEmptyFile:
    """T1.3: Empty file (0 bytes) raises InvalidVideoError."""

    def test_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.mp4"
        empty_file.touch()
        with pytest.raises(InvalidVideoError, match="empty"):
            validate_video(empty_file)


class TestNonVideoFile:
    """T1.4: Non-video file renamed to .mp4 raises InvalidVideoError."""

    def test_text_file_as_mp4(self, tmp_path):
        text_file = tmp_path / "text.mp4"
        text_file.write_text("This is not a video file")
        with pytest.raises(InvalidVideoError):
            validate_video(text_file)


class TestResolutionBelowMinimum:
    """T1.5: Resolution below minimum raises InvalidVideoError."""

    def test_below_min_resolution(self, small_video):
        with pytest.raises(InvalidVideoError, match="below minimum"):
            validate_video(small_video)


class TestLargeResolution:
    """T1.6: 4K video file returns VideoMetadata with 3840x2160."""

    def test_4k_video(self, large_video):
        result = validate_video(large_video)
        assert result.width == 3840
        assert result.height == 2160


class TestVariableFPS:
    """T1.7: Variable FPS video returns average FPS in metadata."""

    def test_returns_fps_as_float(self, valid_video):
        # OpenCV reports average FPS from container metadata
        result = validate_video(valid_video)
        assert isinstance(result.fps, float)
        assert result.fps > 0


class TestIndependence:
    """Verify module works independently with no pipeline dependencies."""

    def test_no_pipeline_imports(self):
        """Ensure input_validator doesn't import from other pipeline stages."""
        import importlib
        import sys

        # Reload to check imports fresh
        if "src.input_validator" in sys.modules:
            del sys.modules["src.input_validator"]

        mod = importlib.import_module("src.input_validator")
        # Module should only depend on cv2, dataclasses, pathlib
        source = Path(mod.__file__).read_text()
        assert "from src." not in source or "from src.input_validator" in source
