"""Unit tests for pose_2d_extractor module.

Tests cover all acceptance criteria and test cases from Task 002:
- T2.1: Single person, clear frame
- T2.2: Multiple people - select highest confidence
- T2.3: No detection in frame - fill with NaN
- T2.4: Full video processing - correct output shape
- T2.5: Output file format - NPZ with 'poses' key
- T2.6: Empty video (0 frames)
- T2.7: Low confidence detection - still stored
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.pose_2d_extractor import (
    _select_primary_detection,
    extract_2d_keypoints,
    load_poses,
)


class TestSelectPrimaryDetection:
    """Tests for the _select_primary_detection helper function."""

    def test_single_detection(self):
        """Test that single detection is returned as-is."""
        keypoints = np.random.rand(1, 17, 3)
        result = _select_primary_detection(keypoints)
        np.testing.assert_array_equal(result, keypoints[0])

    def test_multiple_detections_selects_highest_confidence(self):
        """T2.2: Multiple people - select highest confidence."""
        # Create 3 detections with different confidence levels
        detection1 = np.random.rand(17, 3)
        detection1[:, 2] = 0.3  # Low confidence
        
        detection2 = np.random.rand(17, 3)
        detection2[:, 2] = 0.9  # High confidence
        
        detection3 = np.random.rand(17, 3)
        detection3[:, 2] = 0.5  # Medium confidence
        
        keypoints = np.stack([detection1, detection2, detection3])
        result = _select_primary_detection(keypoints)
        
        # Should select detection2 (highest confidence)
        np.testing.assert_array_equal(result, detection2)


class TestExtract2dKeypoints:
    """Tests for the main extract_2d_keypoints function."""

    @pytest.fixture
    def temp_video_path(self):
        """Create a temporary video path."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)
        path.unlink()  # Remove the file, we just need the path
        yield path
        if path.exists():
            path.unlink()

    def test_nonexistent_video_raises_error(self, temp_output_path):
        """Test that nonexistent video path raises FileNotFoundError."""
        fake_path = Path("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError):
            extract_2d_keypoints(fake_path, temp_output_path, verbose=False)

    @patch('src.pose_2d_extractor.YOLO')
    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_single_person_clear_frame(self, mock_video_capture, mock_yolo, 
                                      temp_video_path, temp_output_path):
        """T2.1: Single person, clear frame - correct 17×3 keypoints."""
        # Setup mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Create mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Single detection with clear keypoints
        keypoints = np.random.rand(1, 17, 3)
        keypoints[:, :, :2] *= 640  # x, y coordinates
        keypoints[:, :, 2] = 0.95  # High confidence
        
        # Setup mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_result = Mock()
        mock_result.keypoints.data.cpu.return_value.numpy.return_value = keypoints
        mock_model.return_value = [mock_result]
        
        # Simulate one frame then end
        mock_cap.read.side_effect = [
            (True, mock_frame),
            (False, None)
        ]
        
        # Create a dummy video file
        temp_video_path.touch()
        
        # Run extraction
        frame_count = extract_2d_keypoints(
            temp_video_path, temp_output_path, verbose=False
        )
        
        assert frame_count == 1
        
        # Verify output file format
        loaded_poses = load_poses(temp_output_path)
        assert loaded_poses.shape == (1, 17, 3)
        np.testing.assert_array_equal(loaded_poses[0], keypoints[0])

    @patch('src.pose_2d_extractor.YOLO')
    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_no_detection_fills_nan(self, mock_video_capture, mock_yolo,
                                   temp_video_path, temp_output_path):
        """T2.3: No detection in frame - row of NaN values."""
        # Setup mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Setup mock YOLO model with no detections
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_result = Mock()
        mock_result.keypoints.data.cpu.return_value.numpy.return_value = np.array([])  # No detections
        mock_model.return_value = [mock_result]
        
        # Simulate one frame then end
        mock_cap.read.side_effect = [
            (True, mock_frame),
            (False, None)
        ]
        
        temp_video_path.touch()
        
        # Run extraction
        frame_count = extract_2d_keypoints(
            temp_video_path, temp_output_path, verbose=False
        )
        
        assert frame_count == 1
        
        # Verify NaN filling
        loaded_poses = load_poses(temp_output_path)
        assert loaded_poses.shape == (1, 17, 3)
        assert np.all(np.isnan(loaded_poses[0]))

    @patch('src.pose_2d_extractor.YOLO')
    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_full_video_processing(self, mock_video_capture, mock_yolo,
                                   temp_video_path, temp_output_path):
        """T2.4: Full video processing - output shape (10, 17, 3)."""
        # Setup mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Setup mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Create consistent keypoints for all frames
        keypoints = np.random.rand(1, 17, 3)
        
        mock_result = Mock()
        mock_result.keypoints.data.cpu.return_value.numpy.return_value = keypoints
        mock_model.return_value = [mock_result]
        
        # Simulate 10 frames
        mock_cap.read.side_effect = (
            [(True, mock_frame)] * 10 + [(False, None)]
        )
        
        temp_video_path.touch()
        
        # Run extraction
        frame_count = extract_2d_keypoints(
            temp_video_path, temp_output_path, verbose=False
        )
        
        assert frame_count == 10
        
        # Verify output shape
        loaded_poses = load_poses(temp_output_path)
        assert loaded_poses.shape == (10, 17, 3)

    @patch('src.pose_2d_extractor.YOLO')
    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_output_file_format(self, mock_video_capture, mock_yolo,
                                temp_video_path, temp_output_path):
        """T2.5: Output file format - NPZ file with 'poses' key."""
        # Setup mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Setup mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        keypoints = np.random.rand(1, 17, 3)
        mock_result = Mock()
        mock_result.keypoints.data.cpu.return_value.numpy.return_value = keypoints
        mock_model.return_value = [mock_result]
        
        mock_cap.read.side_effect = [
            (True, mock_frame),
            (False, None)
        ]
        
        temp_video_path.touch()
        
        # Run extraction
        extract_2d_keypoints(temp_video_path, temp_output_path, verbose=False)
        
        # Verify NPZ format
        assert temp_output_path.exists()
        assert temp_output_path.suffix == '.npz'
        
        # Verify 'poses' key exists
        data = np.load(temp_output_path)
        assert 'poses' in data
        data.close()

    @patch('src.pose_2d_extractor.YOLO')
    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_empty_video(self, mock_video_capture, mock_yolo,
                        temp_video_path, temp_output_path):
        """T2.6: Empty video (0 frames) - returns 0, saves empty array."""
        # Setup mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Setup mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Immediately return False (no frames)
        mock_cap.read.return_value = (False, None)
        
        temp_video_path.touch()
        
        # Run extraction
        frame_count = extract_2d_keypoints(
            temp_video_path, temp_output_path, verbose=False
        )
        
        assert frame_count == 0
        
        # Verify empty array saved
        loaded_poses = load_poses(temp_output_path)
        assert loaded_poses.shape[0] == 0

    @patch('src.pose_2d_extractor.YOLO')
    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_low_confidence_detection_stored(self, mock_video_capture, mock_yolo,
                                            temp_video_path, temp_output_path):
        """T2.7: Low confidence detection - still stored (not filtered)."""
        # Setup mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create low confidence keypoints (all < 0.1)
        keypoints = np.random.rand(1, 17, 3)
        keypoints[:, :, 2] = 0.05  # Very low confidence
        
        # Setup mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_result = Mock()
        mock_result.keypoints.data.cpu.return_value.numpy.return_value = keypoints
        mock_model.return_value = [mock_result]
        
        mock_cap.read.side_effect = [
            (True, mock_frame),
            (False, None)
        ]
        
        temp_video_path.touch()
        
        # Run extraction
        frame_count = extract_2d_keypoints(
            temp_video_path, temp_output_path, verbose=False
        )
        
        assert frame_count == 1
        
        # Verify low confidence data is still stored
        loaded_poses = load_poses(temp_output_path)
        assert loaded_poses.shape == (1, 17, 3)
        # Should not be NaN - should be the actual low-confidence data
        assert not np.all(np.isnan(loaded_poses[0]))
        np.testing.assert_array_equal(loaded_poses[0], keypoints[0])

    @patch('src.pose_2d_extractor.cv2.VideoCapture')
    def test_cannot_open_video_raises_error(self, mock_video_capture,
                                           temp_video_path, temp_output_path):
        """Test that video that cannot be opened raises ValueError."""
        # Mock VideoCapture to return a capture that's not opened
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        temp_video_path.touch()
        
        with pytest.raises(ValueError, match="Could not open video"):
            extract_2d_keypoints(temp_video_path, temp_output_path, verbose=False)


class TestLoadPoses:
    """Tests for the load_poses utility function."""

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent NPZ raises FileNotFoundError."""
        fake_path = Path("/nonexistent/poses.npz")
        with pytest.raises(FileNotFoundError):
            load_poses(fake_path)

    def test_load_npz_without_poses_key_raises_error(self):
        """Test that NPZ without 'poses' key raises KeyError."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)
        
        try:
            # Save NPZ with wrong key
            np.savez(path, wrong_key=np.array([1, 2, 3]))
            
            with pytest.raises(KeyError, match="does not contain 'poses' key"):
                load_poses(path)
        finally:
            if path.exists():
                path.unlink()

    def test_load_valid_npz(self):
        """Test loading valid NPZ file with poses."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)
        
        try:
            # Save valid poses
            expected_poses = np.random.rand(10, 17, 3)
            np.savez(path, poses=expected_poses)
            
            loaded_poses = load_poses(path)
            np.testing.assert_array_equal(loaded_poses, expected_poses)
        finally:
            if path.exists():
                path.unlink()
