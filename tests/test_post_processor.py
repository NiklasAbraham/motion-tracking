"""
Unit tests for post_processor module.

Test cases follow the specification in tasks/005_post_processing.md
"""

import numpy as np
import pytest

from src.post_processor import (
    apply_butterworth_filter,
    enforce_bone_lengths,
    interpolate_missing_frames,
    post_process,
    BONE_LENGTH_CONSTRAINTS,
)


class TestButterworthFilter:
    """Tests for apply_butterworth_filter function (T5.1, T5.5, T5.7)."""
    
    def test_smoothing_reduces_noise(self):
        """T5.1: Smoothing reduces noise - noisy sine wave + random jitter."""
        # Create a clean sine wave
        frames = 200
        t = np.linspace(0, 4 * np.pi, frames)
        clean_signal = np.sin(t)
        
        # Add random jitter
        np.random.seed(42)
        noise = np.random.normal(0, 0.3, frames)
        noisy_signal = clean_signal + noise
        
        # Reshape to (frames, joints, coords) format
        noisy_data = noisy_signal.reshape(frames, 1, 1)
        
        # Apply filter
        smoothed_data = apply_butterworth_filter(noisy_data, cutoff=3.0, fs=30.0, order=2)
        
        # Check that smoothed data is closer to clean signal than noisy data
        smoothed_signal = smoothed_data[:, 0, 0]
        noisy_error = np.mean(np.abs(noisy_signal - clean_signal))
        smoothed_error = np.mean(np.abs(smoothed_signal - clean_signal))
        
        assert smoothed_error < noisy_error, "Smoothed signal should be closer to clean signal"
        assert smoothed_data.shape == noisy_data.shape, "Shape should be preserved"
    
    def test_custom_filter_params(self):
        """T5.5: Custom filter params - cutoff=5, fs=60, order=4."""
        frames = 200
        joints = 16
        coords = 3
        
        # Create test data
        np.random.seed(42)
        data = np.random.randn(frames, joints, coords)
        
        # Apply filter with custom parameters
        filtered = apply_butterworth_filter(data, cutoff=5.0, fs=60.0, order=4)
        
        # Check that filter was applied (data changed) and shape preserved
        assert filtered.shape == data.shape, "Shape should be preserved"
        assert not np.allclose(filtered, data), "Data should be filtered"
    
    def test_single_frame_input(self):
        """T5.7: Single frame input - (1, 16, 3) array returns unchanged."""
        data = np.random.randn(1, 16, 3)
        result = apply_butterworth_filter(data, cutoff=3.0, fs=30.0, order=2)
        
        assert result.shape == (1, 16, 3), "Shape should be preserved"
        assert np.allclose(result, data), "Single frame should be returned unchanged"
    
    def test_shape_preservation_3d(self):
        """Verify shape preservation for 3D input."""
        data = np.random.randn(100, 16, 3)
        result = apply_butterworth_filter(data)
        assert result.shape == data.shape
    
    def test_shape_preservation_2d(self):
        """Verify shape preservation for 2D input."""
        data = np.random.randn(100, 48)
        result = apply_butterworth_filter(data)
        assert result.shape == data.shape


class TestBoneLengthEnforcement:
    """Tests for enforce_bone_lengths function (T5.2)."""
    
    def test_bone_length_enforcement(self):
        """T5.2: Bone-length enforcement - joint pair exceeding max length is clamped."""
        # Create test data with one bone exceeding max length
        frames = 10
        joints = 16
        coords = 3
        
        data = np.zeros((frames, joints, coords))
        
        # Set up two joints (e.g., hip center at index 0 and spine at index 1)
        # According to constraints, (0, 1) should have max_length of 0.35 meters
        data[:, 0, :] = [0, 0, 0]  # Hip center at origin
        data[:, 1, :] = [0.5, 0, 0]  # Spine at 0.5 meters (exceeds 0.35m max)
        
        # Apply bone length enforcement
        result = enforce_bone_lengths(data)
        
        # Check that the bone length is now clamped to max_length
        for frame_idx in range(frames):
            bone_vector = result[frame_idx, 1, :] - result[frame_idx, 0, :]
            bone_length = np.linalg.norm(bone_vector)
            _, max_length = BONE_LENGTH_CONSTRAINTS[(0, 1)]
            
            assert bone_length <= max_length + 1e-6, f"Bone length {bone_length} exceeds max {max_length}"
            assert bone_length >= max_length - 1e-6, f"Bone length should be clamped to max {max_length}"
    
    def test_bone_length_min_enforcement(self):
        """Test that bones shorter than min length are stretched."""
        frames = 5
        joints = 16
        coords = 3
        
        data = np.zeros((frames, joints, coords))
        
        # Set up two joints too close together
        data[:, 0, :] = [0, 0, 0]
        data[:, 1, :] = [0.05, 0, 0]  # Only 0.05m, below min of 0.15m
        
        result = enforce_bone_lengths(data)
        
        # Check that the bone length is now at least min_length
        for frame_idx in range(frames):
            bone_vector = result[frame_idx, 1, :] - result[frame_idx, 0, :]
            bone_length = np.linalg.norm(bone_vector)
            min_length, _ = BONE_LENGTH_CONSTRAINTS[(0, 1)]
            
            assert bone_length >= min_length - 1e-6, f"Bone length {bone_length} below min {min_length}"
    
    def test_shape_preservation(self):
        """Verify shape preservation."""
        data = np.random.randn(100, 16, 3)
        result = enforce_bone_lengths(data)
        assert result.shape == data.shape


class TestInterpolateMissingFrames:
    """Tests for interpolate_missing_frames function (T5.3, T5.6)."""
    
    def test_nan_interpolation(self):
        """T5.3: NaN interpolation - 3 consecutive NaN frames are interpolated."""
        frames = 10
        joints = 16
        coords = 3
        
        # Create data with some NaN frames
        data = np.ones((frames, joints, coords))
        
        # Insert 3 consecutive NaN frames in the middle (frames 4, 5, 6)
        data[4:7, 0, 0] = np.nan
        
        # Apply interpolation
        result = interpolate_missing_frames(data)
        
        # Check that NaN values were filled
        assert not np.any(np.isnan(result[4:7, 0, 0])), "NaN values should be interpolated"
        
        # Check that interpolated values are reasonable (should be close to 1.0)
        assert np.allclose(result[4:7, 0, 0], 1.0, atol=0.1), "Interpolated values should be close to surrounding values"
    
    def test_all_nan_input(self):
        """T5.6: All-NaN input - entire array is NaN raises ValueError."""
        data = np.full((10, 16, 3), np.nan)
        
        with pytest.raises(ValueError, match="Cannot interpolate: all frames are NaN"):
            interpolate_missing_frames(data)
    
    def test_single_valid_point(self):
        """Test interpolation with only one valid point."""
        data = np.full((10, 2, 3), np.nan)
        data[5, 0, 0] = 1.0
        
        result = interpolate_missing_frames(data)
        
        # All frames for this joint/coord should be filled with the single valid value
        assert np.allclose(result[:, 0, 0], 1.0), "Should fill with single valid value"
    
    def test_shape_preservation(self):
        """Verify shape preservation."""
        data = np.random.randn(100, 16, 3)
        # Add some NaN values
        data[10:15, 5, 1] = np.nan
        
        result = interpolate_missing_frames(data)
        assert result.shape == data.shape


class TestPostProcess:
    """Tests for complete post_process pipeline (T5.4)."""
    
    def test_shape_preservation(self):
        """T5.4: Shape preservation - (200, 16, 3) input produces (200, 16, 3) output."""
        frames = 200
        joints = 16
        coords = 3
        
        data = np.random.randn(frames, joints, coords)
        result = post_process(data, cutoff=3.0, fs=30.0, order=2)
        
        assert result.shape == (frames, joints, coords), "Output shape should match input shape"
    
    def test_complete_pipeline(self):
        """Test that the complete pipeline runs without errors."""
        # Create realistic test data with some issues
        frames = 100
        joints = 16
        coords = 3
        
        np.random.seed(42)
        data = np.random.randn(frames, joints, coords) * 0.5
        
        # Add some NaN values
        data[10:13, 5, 1] = np.nan
        
        # Add some bone length violations
        data[:, 0, :] = 0
        data[:, 1, :] = [0.6, 0, 0]  # Exceeds max length
        
        # Run complete pipeline
        result = post_process(data, cutoff=3.0, fs=30.0, order=2)
        
        # Check results
        assert result.shape == data.shape, "Shape should be preserved"
        assert not np.any(np.isnan(result[10:13, 5, 1])), "NaN values should be interpolated"
        
        # Check bone length constraint
        bone_length = np.linalg.norm(result[0, 1, :] - result[0, 0, :])
        _, max_length = BONE_LENGTH_CONSTRAINTS[(0, 1)]
        assert bone_length <= max_length + 1e-5, "Bone length should be enforced"
    
    def test_invalid_shape_raises_error(self):
        """Test that invalid input shape raises ValueError."""
        # 2D input should raise error
        data = np.random.randn(100, 48)
        
        with pytest.raises(ValueError, match="Expected 3D pose array"):
            post_process(data)
    
    def test_optional_steps(self):
        """Test that optional processing steps can be disabled."""
        data = np.random.randn(50, 16, 3)
        data[5:8, 0, 0] = np.nan
        
        # Disable interpolation and constraints
        result = post_process(data, interpolate=False, enforce_constraints=False)
        
        assert result.shape == data.shape, "Shape should be preserved"
        # NaN values should still be present if interpolation is disabled
        assert np.any(np.isnan(result[5:8, 0, 0])), "NaN values should remain when interpolation is disabled"


class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_empty_frames(self):
        """Test handling of empty input."""
        data = np.random.randn(0, 16, 3)
        
        # Should handle gracefully
        result = apply_butterworth_filter(data)
        assert result.shape == data.shape
    
    def test_two_frames(self):
        """Test with minimum frames for filtering."""
        data = np.random.randn(2, 16, 3)
        result = apply_butterworth_filter(data)
        assert result.shape == data.shape
    
    def test_nan_in_filter(self):
        """Test that filter handles NaN values gracefully."""
        data = np.random.randn(50, 16, 3)
        data[10:15, 5, 1] = np.nan
        
        result = apply_butterworth_filter(data)
        assert result.shape == data.shape
        # NaN values should still be NaN after filtering
        assert np.all(np.isnan(result[10:15, 5, 1]))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
