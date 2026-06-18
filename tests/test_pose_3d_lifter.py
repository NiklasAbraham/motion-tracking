"""
Unit tests for 3D pose lifting module.

Tests use mocked DiffPose inference to validate module functionality
without requiring actual model files.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pose_3d_lifter import lift_2d_to_3d, LiftingConfig, ConfigError, load_diffpose_model


class TestPose3DLifter:
    """Test suite for pose_3d_lifter module."""
    
    # Threshold for temporal consistency validation in mocked implementation.
    # This value is appropriate because the mock generates synthetic depth values
    # that vary smoothly, and frame-to-frame differences in smooth sinusoidal
    # motion should be well under 1.0 units.
    TEMPORAL_CONSISTENCY_THRESHOLD = 1.0
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        tmp = tempfile.mkdtemp()
        yield Path(tmp)
        shutil.rmtree(tmp)
    
    @pytest.fixture
    def mock_checkpoint(self, temp_dir):
        """Create a mock checkpoint file."""
        checkpoint_path = temp_dir / "mock_checkpoint.pth"
        checkpoint_path.write_text("mock checkpoint data")
        return checkpoint_path
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock config file."""
        config_path = temp_dir / "mock_config.yml"
        config_path.write_text("mock config data")
        return config_path
    
    def test_T4_1_mocked_lifting(self, temp_dir, mock_checkpoint):
        """T4.1: Test mocked lifting with synthetic (50, 16, 2) input."""
        # Create synthetic input
        input_2d = np.random.randn(50, 16, 2).astype(np.float32)
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        # Create config
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        # Run lifting
        num_frames = lift_2d_to_3d(input_path, output_path, config)
        
        # Verify
        assert num_frames == 50
        assert output_path.exists()
        
        output_3d = np.load(output_path)
        assert output_3d.shape == (50, 16, 3)
        assert output_3d.dtype == np.float32
    
    def test_T4_2_missing_checkpoint(self, temp_dir):
        """T4.2: Test missing checkpoint raises FileNotFoundError."""
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        # Create minimal input
        input_2d = np.zeros((10, 16, 2), dtype=np.float32)
        np.save(input_path, input_2d)
        
        # Config with non-existent checkpoint
        config = LiftingConfig(checkpoint_path=temp_dir / "nonexistent.pth")
        
        # Should raise FileNotFoundError with informative message
        with pytest.raises(FileNotFoundError) as exc_info:
            lift_2d_to_3d(input_path, output_path, config)
        
        assert "checkpoint not found" in str(exc_info.value).lower()
    
    def test_T4_3_invalid_config(self, temp_dir, mock_checkpoint):
        """T4.3: Test invalid config path raises ConfigError."""
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        # Create minimal input
        input_2d = np.zeros((10, 16, 2), dtype=np.float32)
        np.save(input_path, input_2d)
        
        # Config with non-existent config file
        config = LiftingConfig(
            checkpoint_path=mock_checkpoint,
            config_path=temp_dir / "nonexistent.yml"
        )
        
        # Should raise ConfigError
        with pytest.raises(ConfigError) as exc_info:
            lift_2d_to_3d(input_path, output_path, config)
        
        assert "configuration file not found" in str(exc_info.value).lower()
    
    def test_T4_4_output_file_creation(self, temp_dir, mock_checkpoint):
        """T4.4: Test output NPY file is created on disk."""
        # Create input
        input_2d = np.random.randn(25, 16, 2).astype(np.float32)
        input_path = temp_dir / "input_2d.npy"
        
        # Output in nested directory
        output_dir = temp_dir / "outputs" / "nested"
        output_path = output_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        # Config
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        # Run lifting
        lift_2d_to_3d(input_path, output_path, config)
        
        # Verify file exists and nested directories were created
        assert output_path.exists()
        assert output_path.is_file()
        
        # Verify content is valid
        output_3d = np.load(output_path)
        assert output_3d.shape == (25, 16, 3)
    
    def test_T4_5_zero_frame_input(self, temp_dir, mock_checkpoint):
        """T4.5: Test zero-frame input (0, 16, 2) returns (0, 16, 3)."""
        # Create empty input
        input_2d = np.zeros((0, 16, 2), dtype=np.float32)
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        # Config
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        # Run lifting
        num_frames = lift_2d_to_3d(input_path, output_path, config)
        
        # Verify
        assert num_frames == 0
        assert output_path.exists()
        
        output_3d = np.load(output_path)
        assert output_3d.shape == (0, 16, 3)
    
    def test_T4_6_temporal_consistency(self, temp_dir, mock_checkpoint):
        """T4.6: Test smooth 2D input produces temporally consistent 3D output."""
        # Create smooth temporal sequence
        num_frames = 100
        t = np.linspace(0, 2 * np.pi, num_frames)
        
        input_2d = np.zeros((num_frames, 16, 2), dtype=np.float32)
        
        # Create smooth sinusoidal motion for all joints
        # Phase offset (0.1) ensures each joint has different but smooth motion patterns
        for joint_idx in range(16):
            input_2d[:, joint_idx, 0] = np.sin(t + joint_idx * 0.1)
            input_2d[:, joint_idx, 1] = np.cos(t + joint_idx * 0.1)
        
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        # Config
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        # Run lifting
        lift_2d_to_3d(input_path, output_path, config)
        
        # Load output
        output_3d = np.load(output_path)
        
        # Check temporal consistency: frame-to-frame differences should be small
        frame_diffs = np.diff(output_3d, axis=0)
        max_diff = np.max(np.abs(frame_diffs))
        
        # Verify temporal consistency using class-level threshold
        assert max_diff < self.TEMPORAL_CONSISTENCY_THRESHOLD, \
            f"Max frame diff {max_diff} exceeds threshold {self.TEMPORAL_CONSISTENCY_THRESHOLD}"
    
    def test_T4_7_nan_handling(self, temp_dir, mock_checkpoint):
        """T4.7: Test frames with NaN remain NaN in output."""
        # Create input with some NaN frames
        input_2d = np.random.randn(50, 16, 2).astype(np.float32)
        
        # Set specific frames to NaN
        nan_frame_indices = [5, 15, 25, 40]
        for idx in nan_frame_indices:
            input_2d[idx, :, :] = np.nan
        
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        # Config
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        # Run lifting
        lift_2d_to_3d(input_path, output_path, config)
        
        # Load output
        output_3d = np.load(output_path)
        
        # Verify NaN frames are preserved
        for idx in nan_frame_indices:
            assert np.isnan(output_3d[idx]).all(), f"Frame {idx} should be all NaN"
        
        # Verify non-NaN frames are not NaN
        for idx in range(50):
            if idx not in nan_frame_indices:
                assert not np.isnan(output_3d[idx]).any(), f"Frame {idx} should not have NaN"
    
    def test_invalid_input_shape_ndim(self, temp_dir, mock_checkpoint):
        """Test that 2D input (not 3D) raises ValueError."""
        # Create 2D input (incorrect)
        input_2d = np.random.randn(16, 2).astype(np.float32)
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        with pytest.raises(ValueError) as exc_info:
            lift_2d_to_3d(input_path, output_path, config)
        
        assert "expected 3d input array" in str(exc_info.value).lower()
    
    def test_invalid_joint_count(self, temp_dir, mock_checkpoint):
        """Test that input with wrong joint count raises ValueError."""
        # Create input with 17 joints (YOLO format, not DiffPose)
        input_2d = np.random.randn(50, 17, 2).astype(np.float32)
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        with pytest.raises(ValueError) as exc_info:
            lift_2d_to_3d(input_path, output_path, config)
        
        assert "expected 16 joints" in str(exc_info.value).lower()
    
    def test_invalid_coordinate_dims(self, temp_dir, mock_checkpoint):
        """Test that input with 3D coordinates (not 2D) raises ValueError."""
        # Create input with 3D coordinates
        input_2d = np.random.randn(50, 16, 3).astype(np.float32)
        input_path = temp_dir / "input_2d.npy"
        output_path = temp_dir / "output_3d.npy"
        
        np.save(input_path, input_2d)
        
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        
        with pytest.raises(ValueError) as exc_info:
            lift_2d_to_3d(input_path, output_path, config)
        
        assert "expected 2d coordinates" in str(exc_info.value).lower()
    
    def test_load_diffpose_model(self, mock_checkpoint):
        """Test load_diffpose_model function."""
        config = LiftingConfig(checkpoint_path=mock_checkpoint)
        model = load_diffpose_model(config)
        
        # Mock model should have checkpoint info
        assert model is not None
        assert "checkpoint" in model
        assert "loaded" in model
    
    def test_load_diffpose_model_missing_checkpoint(self, temp_dir):
        """Test load_diffpose_model raises error for missing checkpoint."""
        config = LiftingConfig(checkpoint_path=temp_dir / "nonexistent.pth")
        
        with pytest.raises(FileNotFoundError):
            load_diffpose_model(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
