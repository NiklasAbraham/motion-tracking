"""Tests for keypoint_mapper module.

Test cases cover:
- T3.1: Basic mapping with known input
- T3.2: Multi-frame mapping
- T3.3: NaN frame passthrough
- T3.4: Joint index correctness
- T3.5: Confidence stripping
- T3.6: Zero-frame input
- T3.7: Invalid input shape
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keypoint_mapper import map_yolo_to_diffpose, YOLO_TO_DIFFPOSE_MAPPING


class TestKeypointMapper:
    """Test suite for YOLO to DiffPose keypoint mapping."""
    
    def test_basic_mapping(self):
        """T3.1: Basic mapping with known input."""
        # Create a single frame with known values
        yolo_input = np.zeros((1, 17, 3))
        # Set specific values for tracked joints
        yolo_input[0, 0, :] = [100, 200, 0.9]  # Nose
        yolo_input[0, 5, :] = [50, 100, 0.8]   # Left Shoulder
        yolo_input[0, 12, :] = [150, 250, 0.85] # Right Hip
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Check output shape
        assert result.shape == (1, 16, 2), f"Expected shape (1, 16, 2), got {result.shape}"
        
        # Check that nose (YOLO 0) maps to head (DiffPose 8)
        np.testing.assert_array_almost_equal(result[0, 8, :], [100, 200])
        
        # Check that left shoulder (YOLO 5) maps to DiffPose 10
        np.testing.assert_array_almost_equal(result[0, 10, :], [50, 100])
        
        # Check that right hip (YOLO 12) maps to DiffPose 0
        np.testing.assert_array_almost_equal(result[0, 0, :], [150, 250])
    
    def test_multi_frame_mapping(self):
        """T3.2: Multi-frame mapping."""
        frames = 100
        yolo_input = np.random.rand(frames, 17, 3) * 1000
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Check output shape
        assert result.shape == (frames, 16, 2), \
            f"Expected shape ({frames}, 16, 2), got {result.shape}"
        
        # Verify that each YOLO joint maps to correct DiffPose joint for all frames
        for yolo_idx, diffpose_idx in YOLO_TO_DIFFPOSE_MAPPING.items():
            np.testing.assert_array_almost_equal(
                result[:, diffpose_idx, :],
                yolo_input[:, yolo_idx, :2],
                err_msg=f"YOLO joint {yolo_idx} did not map correctly to DiffPose joint {diffpose_idx}"
            )
    
    def test_nan_frame_passthrough(self):
        """T3.3: NaN frame passthrough."""
        # Create input with one frame of all NaN
        yolo_input = np.full((3, 17, 3), np.nan)
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Check that output is also all NaN
        assert result.shape == (3, 16, 2)
        assert np.all(np.isnan(result)), "Expected all NaN output for all NaN input"
    
    def test_partial_nan_handling(self):
        """Test handling of frames with some NaN values."""
        yolo_input = np.random.rand(5, 17, 3) * 1000
        # Set some joints to NaN
        yolo_input[2, 0, :] = np.nan  # Frame 2, nose
        yolo_input[3, 5:8, :] = np.nan  # Frame 3, some joints
        
        result = map_yolo_to_diffpose(yolo_input)
        
        assert result.shape == (5, 16, 2)
        # Check that NaN values are preserved
        assert np.isnan(result[2, 8, 0]), "NaN should be preserved for nose mapping"
        assert np.isnan(result[3, 10, 0]), "NaN should be preserved for left shoulder"
    
    def test_joint_index_correctness(self):
        """T3.4: Joint index correctness - verify left shoulder mapping."""
        yolo_input = np.zeros((1, 17, 3))
        # Set unique values for left shoulder (YOLO index 5)
        yolo_input[0, 5, :] = [123.45, 678.90, 0.95]
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Left shoulder at YOLO[5] should map to DiffPose[10]
        # (spec says 11 but 0-indexed is 10)
        expected_diffpose_idx = YOLO_TO_DIFFPOSE_MAPPING[5]
        assert expected_diffpose_idx == 10, "Mapping constant should map YOLO 5 to DiffPose 10"
        np.testing.assert_array_almost_equal(
            result[0, 10, :],
            [123.45, 678.90],
            err_msg="Left shoulder should map from YOLO[5] to DiffPose[10]"
        )
    
    def test_confidence_stripping(self):
        """T3.5: Confidence stripping - input has confidence dimension."""
        yolo_input = np.random.rand(10, 17, 3)
        # Set confidence values
        yolo_input[:, :, 2] = np.random.rand(10, 17) * 0.5 + 0.5  # confidence 0.5-1.0
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Output should only have x, y (no confidence)
        assert result.shape == (10, 16, 2), "Output should have only 2 coordinates per joint"
        
        # Verify that confidence is not in output
        for yolo_idx, diffpose_idx in YOLO_TO_DIFFPOSE_MAPPING.items():
            np.testing.assert_array_almost_equal(
                result[:, diffpose_idx, :],
                yolo_input[:, yolo_idx, :2],  # Only x, y
                err_msg=f"Output should only contain x, y coordinates (no confidence)"
            )
    
    def test_two_dimensional_input(self):
        """Test that input with only x, y (no confidence) also works."""
        yolo_input = np.random.rand(10, 17, 2) * 1000
        
        result = map_yolo_to_diffpose(yolo_input)
        
        assert result.shape == (10, 16, 2)
        
        # Verify mapping is still correct
        for yolo_idx, diffpose_idx in YOLO_TO_DIFFPOSE_MAPPING.items():
            np.testing.assert_array_almost_equal(
                result[:, diffpose_idx, :],
                yolo_input[:, yolo_idx, :],
                err_msg=f"Mapping should work with 2D input (no confidence)"
            )
    
    def test_zero_frame_input(self):
        """T3.6: Zero-frame input."""
        yolo_input = np.zeros((0, 17, 3))
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Should return empty array with correct shape
        assert result.shape == (0, 16, 2), \
            f"Expected shape (0, 16, 2) for zero frames, got {result.shape}"
    
    def test_invalid_input_shape_wrong_joints(self):
        """T3.7: Invalid input shape - wrong number of joints."""
        # Input with 15 joints instead of 17
        yolo_input = np.random.rand(10, 15, 3)
        
        with pytest.raises(ValueError, match="Expected 17 joints"):
            map_yolo_to_diffpose(yolo_input)
    
    def test_invalid_input_shape_wrong_dimensions(self):
        """T3.7: Invalid input shape - wrong number of dimensions."""
        # 2D input instead of 3D
        yolo_input = np.random.rand(17, 3)
        
        with pytest.raises(ValueError, match="Expected 3D array"):
            map_yolo_to_diffpose(yolo_input)
    
    def test_invalid_input_shape_wrong_coords(self):
        """T3.7: Invalid input shape - wrong coordinate dimensions."""
        # 4 coordinate dimensions instead of 2 or 3
        yolo_input = np.random.rand(10, 17, 4)
        
        with pytest.raises(ValueError, match="Expected 2 or 3 coordinate dimensions"):
            map_yolo_to_diffpose(yolo_input)
    
    def test_all_joints_mapped(self):
        """Verify that all 13 YOLO joints are mapped (13 joints used, not all 17)."""
        # Check that mapping has correct number of entries
        assert len(YOLO_TO_DIFFPOSE_MAPPING) == 13, \
            "Should map 13 YOLO joints (nose + 6 per side for arms and legs)"
        
        # Verify all DiffPose indices 0-15 are covered
        diffpose_indices = set(YOLO_TO_DIFFPOSE_MAPPING.values())
        # We expect some indices to be used and some to potentially be NaN
        assert len(diffpose_indices) == 13, "Should map to 13 unique DiffPose joints"
        
        # All DiffPose indices should be in range [0, 15]
        assert all(0 <= idx <= 15 for idx in diffpose_indices), \
            "All DiffPose indices should be in range [0, 15]"
    
    def test_unmapped_joints_are_nan(self):
        """Verify that unmapped DiffPose joints remain NaN."""
        yolo_input = np.random.rand(1, 17, 3) * 1000
        
        result = map_yolo_to_diffpose(yolo_input)
        
        # Find which DiffPose indices are not in the mapping
        mapped_diffpose_indices = set(YOLO_TO_DIFFPOSE_MAPPING.values())
        all_diffpose_indices = set(range(16))
        unmapped_indices = all_diffpose_indices - mapped_diffpose_indices
        
        # Check that unmapped joints are NaN
        for idx in unmapped_indices:
            assert np.all(np.isnan(result[0, idx, :])), \
                f"DiffPose joint {idx} should be NaN (unmapped)"
    
    def test_dtype_preservation(self):
        """Test that output preserves input dtype."""
        for dtype in [np.float32, np.float64]:
            yolo_input = np.random.rand(5, 17, 3).astype(dtype)
            result = map_yolo_to_diffpose(yolo_input)
            assert result.dtype == dtype, \
                f"Output dtype {result.dtype} should match input dtype {dtype}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
