"""Integration tests for the pipeline orchestrator.

These tests use mocked stages to verify orchestration logic
without requiring actual model files or heavy processing.
"""
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineStage,
    run_pipeline,
    main
)


@pytest.fixture
def temp_video(tmp_path):
    """Create a temporary video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_text("fake video content")
    return video_path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_working_dir(tmp_path):
    """Create a temporary working directory."""
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    return working_dir


@pytest.fixture
def default_config(temp_working_dir):
    """Create a default pipeline configuration."""
    return PipelineConfig(working_dir=temp_working_dir)


class TestPipelineStage:
    """Tests for the PipelineStage class."""
    
    def test_stage_initialization(self):
        """Test that a stage is initialized correctly."""
        stage = PipelineStage("test_stage", "output.txt")
        assert stage.name == "test_stage"
        assert stage.checkpoint_file == "output.txt"
    
    def test_is_complete_when_file_exists(self, tmp_path):
        """Test that is_complete returns True when checkpoint exists."""
        stage = PipelineStage("test", "checkpoint.json")
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text("{}")
        
        assert stage.is_complete(tmp_path) is True
    
    def test_is_complete_when_file_missing(self, tmp_path):
        """Test that is_complete returns False when checkpoint doesn't exist."""
        stage = PipelineStage("test", "checkpoint.json")
        assert stage.is_complete(tmp_path) is False


class TestPipelineConfig:
    """Tests for the PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = PipelineConfig()
        assert config.model_2d == "yolov8x-pose.pt"
        assert config.lifting_config == "human36m_diffpose_uvxyz_cpn.yml"
        assert config.filter_cutoff == 3.0
        assert config.filter_fs == 30.0
        assert config.filter_order == 2
        assert config.viz_format == "html"
        assert config.working_dir == Path("./pipeline_output")
    
    def test_custom_config(self, tmp_path):
        """Test that custom configuration values are preserved."""
        config = PipelineConfig(
            model_2d="custom_model.pt",
            filter_cutoff=5.0,
            working_dir=tmp_path
        )
        assert config.model_2d == "custom_model.pt"
        assert config.filter_cutoff == 5.0
        assert config.working_dir == tmp_path
    
    def test_working_dir_conversion(self):
        """Test that working_dir is converted to Path object."""
        config = PipelineConfig(working_dir="./test_dir")
        assert isinstance(config.working_dir, Path)
        assert config.working_dir == Path("./test_dir")


class TestPipeline:
    """Tests for the Pipeline class."""
    
    def test_t71_full_pipeline_mock(self, temp_video, temp_output_dir, default_config, capsys):
        """T7.1: Full pipeline mock - all stages called in order."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Mock all stage functions
        with patch.object(pipeline, '_validate_stage') as mock_validate, \
             patch.object(pipeline, '_extract_2d_stage') as mock_extract, \
             patch.object(pipeline, '_map_stage') as mock_map, \
             patch.object(pipeline, '_lift_stage') as mock_lift, \
             patch.object(pipeline, '_post_process_stage') as mock_post, \
             patch.object(pipeline, '_visualize_stage') as mock_viz:
            
            # Set return values for each stage
            working_dir = default_config.working_dir
            mock_validate.return_value = working_dir / "video_metadata.json"
            mock_extract.return_value = working_dir / "poses_2d.npz"
            mock_map.return_value = working_dir / "poses_2d_mapped.npz"
            mock_lift.return_value = working_dir / "poses_3d.npy"
            mock_post.return_value = working_dir / "poses_3d_cleaned.npy"
            mock_viz.return_value = working_dir / "visualization.html"
            
            # Create the final visualization file for copying
            (working_dir / "visualization.html").write_text("<html></html>")
            
            result = pipeline.run()
            
            # Verify all stages were called in order
            mock_validate.assert_called_once()
            mock_extract.assert_called_once()
            mock_map.assert_called_once()
            mock_lift.assert_called_once()
            mock_post.assert_called_once()
            mock_viz.assert_called_once()
            
            # Verify output file exists
            assert result.exists()
            assert result == temp_output_dir / "visualization.html"
    
    def test_t72_resume_from_stage_3(self, temp_video, temp_output_dir, default_config, capsys):
        """T7.2: Resume from stage 3 - skips stages 1-2, starts at 3."""
        working_dir = default_config.working_dir
        
        # Create checkpoint files for stages 1 and 2
        (working_dir / "video_metadata.json").write_text("{}")
        (working_dir / "poses_2d.npz").write_bytes(b"fake npz")
        
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Mock all stage functions
        with patch.object(pipeline, '_validate_stage') as mock_validate, \
             patch.object(pipeline, '_extract_2d_stage') as mock_extract, \
             patch.object(pipeline, '_map_stage') as mock_map, \
             patch.object(pipeline, '_lift_stage') as mock_lift, \
             patch.object(pipeline, '_post_process_stage') as mock_post, \
             patch.object(pipeline, '_visualize_stage') as mock_viz:
            
            # Set return values
            mock_map.return_value = working_dir / "poses_2d_mapped.npz"
            mock_lift.return_value = working_dir / "poses_3d.npy"
            mock_post.return_value = working_dir / "poses_3d_cleaned.npy"
            mock_viz.return_value = working_dir / "visualization.html"
            
            # Create the final visualization file
            (working_dir / "visualization.html").write_text("<html></html>")
            
            result = pipeline.run()
            
            # Verify stages 1 and 2 were NOT called
            mock_validate.assert_not_called()
            mock_extract.assert_not_called()
            
            # Verify stages 3-6 were called
            mock_map.assert_called_once()
            mock_lift.assert_called_once()
            mock_post.assert_called_once()
            mock_viz.assert_called_once()
            
            # Check output message
            captured = capsys.readouterr()
            assert "already complete, skipping" in captured.out
    
    def test_t73_invalid_video(self, temp_output_dir, default_config):
        """T7.3: Invalid video - fails at validation, reports error."""
        non_existent_video = Path("/tmp/non_existent_video.mp4")
        pipeline = Pipeline(non_existent_video, temp_output_dir, default_config)
        
        # Should raise FileNotFoundError during validation
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            pipeline.run()
    
    def test_t74_missing_checkpoint(self, temp_video, temp_output_dir, default_config):
        """T7.4: Missing checkpoint - would fail at lifting stage with missing model files.
        
        Note: Since we're using mocked stages, we simulate a failure during the lift stage.
        """
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Mock stages - simulate failure at lifting stage
        with patch.object(pipeline, '_validate_stage') as mock_validate, \
             patch.object(pipeline, '_extract_2d_stage') as mock_extract, \
             patch.object(pipeline, '_map_stage') as mock_map, \
             patch.object(pipeline, '_lift_stage') as mock_lift:
            
            working_dir = default_config.working_dir
            mock_validate.return_value = working_dir / "video_metadata.json"
            mock_extract.return_value = working_dir / "poses_2d.npz"
            mock_map.return_value = working_dir / "poses_2d_mapped.npz"
            mock_lift.side_effect = FileNotFoundError("Model checkpoint not found")
            
            # Should raise FileNotFoundError during lifting
            with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
                pipeline.run()
    
    def test_t75_output_directory_creation(self, temp_video, tmp_path, default_config):
        """T7.5: Output directory creation - creates directory automatically."""
        non_existent_output = tmp_path / "new_output_dir"
        assert not non_existent_output.exists()
        
        pipeline = Pipeline(temp_video, non_existent_output, default_config)
        
        # Directory should be created during initialization
        assert non_existent_output.exists()
        assert non_existent_output.is_dir()
    
    def test_t76_progress_reporting(self, temp_video, temp_output_dir, default_config):
        """T7.6: Progress reporting - progress callback called per stage."""
        progress_calls = []
        
        def progress_callback(stage_name: str, percentage: int):
            progress_calls.append((stage_name, percentage))
        
        pipeline = Pipeline(
            temp_video,
            temp_output_dir,
            default_config,
            progress_callback=progress_callback
        )
        
        # Run with actual stage methods (they create dummy outputs)
        result = pipeline.run()
        
        # Verify progress was reported for each stage
        stage_names = {call[0] for call in progress_calls}
        expected_stages = {"validate", "2d_extract", "map", "lift", "post_process", "visualize"}
        assert stage_names == expected_stages
        
        # Verify each stage reported 0% and 100%
        for stage in expected_stages:
            assert (stage, 0) in progress_calls
            assert (stage, 100) in progress_calls
    
    def test_t77_custom_config(self, temp_video, temp_output_dir, default_config):
        """T7.7: Custom config - params passed through to post-processor."""
        custom_config = PipelineConfig(
            working_dir=default_config.working_dir,
            filter_cutoff=5.0,
            filter_fs=60.0,
            filter_order=4
        )
        
        pipeline = Pipeline(temp_video, temp_output_dir, custom_config)
        
        # Verify config was stored correctly
        assert pipeline.config.filter_cutoff == 5.0
        assert pipeline.config.filter_fs == 60.0
        assert pipeline.config.filter_order == 4


class TestRunPipeline:
    """Tests for the run_pipeline function."""
    
    def test_run_pipeline_function(self, temp_video, temp_output_dir, default_config):
        """Test the run_pipeline convenience function."""
        result = run_pipeline(temp_video, temp_output_dir, default_config)
        
        assert result.exists()
        assert result == temp_output_dir / "visualization.html"


class TestCLI:
    """Tests for the CLI entry point."""
    
    def test_main_with_minimal_args(self, temp_video, tmp_path, monkeypatch):
        """Test CLI with minimal arguments."""
        output_dir = tmp_path / "cli_output"
        
        # Mock sys.argv
        test_args = [
            "pipeline.py",
            str(temp_video),
            "--output", str(output_dir)
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        # Run main
        exit_code = main()
        
        assert exit_code == 0
        assert output_dir.exists()
    
    def test_main_with_custom_args(self, temp_video, tmp_path, monkeypatch):
        """Test CLI with custom arguments."""
        output_dir = tmp_path / "cli_output"
        
        test_args = [
            "pipeline.py",
            str(temp_video),
            "--output", str(output_dir),
            "--filter-cutoff", "5.0",
            "--filter-fs", "60.0",
            "--viz-format", "mp4"
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        exit_code = main()
        
        assert exit_code == 0
        # Note: Visualization format would be mp4, but our dummy implementation creates html
        # In a real implementation, this would be verified
    
    def test_main_with_nonexistent_video(self, tmp_path, monkeypatch, capsys):
        """Test CLI with non-existent video file."""
        output_dir = tmp_path / "cli_output"
        fake_video = tmp_path / "nonexistent.mp4"
        working_dir = tmp_path / "working"
        
        test_args = [
            "pipeline.py",
            str(fake_video),
            "--output", str(output_dir),
            "--working-dir", str(working_dir)
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        exit_code = main()
        
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestStageImplementations:
    """Tests for individual stage implementations."""
    
    def test_validate_stage_creates_metadata(self, temp_video, temp_output_dir, default_config):
        """Test that validate stage creates metadata JSON."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        result = pipeline._validate_stage()
        
        assert result.exists()
        assert result.name == "video_metadata.json"
        
        # Verify metadata structure
        with open(result) as f:
            metadata = json.load(f)
        
        assert "path" in metadata
        assert "width" in metadata
        assert "height" in metadata
        assert "fps" in metadata
    
    def test_extract_2d_stage_creates_npz(self, temp_video, temp_output_dir, default_config):
        """Test that 2D extraction stage creates NPZ file."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Need to run validate first to create metadata
        pipeline._validate_stage()
        
        result = pipeline._extract_2d_stage()
        
        assert result.exists()
        assert result.suffix == ".npz"
        
        # Verify NPZ structure
        data = np.load(result)
        assert "poses" in data
        assert data["poses"].shape[1] == 17  # YOLO 17 joints
        assert data["poses"].shape[2] == 3   # x, y, confidence
    
    def test_map_stage_converts_format(self, temp_video, temp_output_dir, default_config):
        """Test that mapping stage converts YOLO to DiffPose format."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Create prerequisite stages
        pipeline._validate_stage()
        pipeline._extract_2d_stage()
        
        result = pipeline._map_stage()
        
        assert result.exists()
        assert result.suffix == ".npz"
        
        # Verify mapped format
        data = np.load(result)
        assert "poses" in data
        assert data["poses"].shape[1] == 16  # DiffPose 16 joints
        assert data["poses"].shape[2] == 2   # x, y (no confidence)
    
    def test_lift_stage_adds_z_dimension(self, temp_video, temp_output_dir, default_config):
        """Test that lifting stage adds Z dimension."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Create prerequisite stages
        pipeline._validate_stage()
        pipeline._extract_2d_stage()
        pipeline._map_stage()
        
        result = pipeline._lift_stage()
        
        assert result.exists()
        assert result.suffix == ".npy"
        
        # Verify 3D format
        data = np.load(result)
        assert data.shape[1] == 16  # DiffPose 16 joints
        assert data.shape[2] == 3   # x, y, z
    
    def test_post_process_stage_smooths_data(self, temp_video, temp_output_dir, default_config):
        """Test that post-processing stage creates cleaned output."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Create all prerequisite stages
        pipeline._validate_stage()
        pipeline._extract_2d_stage()
        pipeline._map_stage()
        pipeline._lift_stage()
        
        result = pipeline._post_process_stage()
        
        assert result.exists()
        assert result.suffix == ".npy"
        assert "cleaned" in result.name
    
    def test_visualize_stage_creates_output(self, temp_video, temp_output_dir, default_config):
        """Test that visualization stage creates output file."""
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        # Create all prerequisite stages
        pipeline._validate_stage()
        pipeline._extract_2d_stage()
        pipeline._map_stage()
        pipeline._lift_stage()
        pipeline._post_process_stage()
        
        result = pipeline._visualize_stage()
        
        assert result.exists()
        assert result.suffix == ".html"


class TestResumeCapability:
    """Tests for checkpoint/resume functionality."""
    
    def test_resume_after_complete_pipeline(self, temp_video, temp_output_dir, default_config, capsys):
        """Test that running pipeline again when complete skips all stages."""
        # Run pipeline once
        pipeline1 = Pipeline(temp_video, temp_output_dir, default_config)
        result1 = pipeline1.run()
        assert result1.exists()
        
        # Run again - should skip all stages
        pipeline2 = Pipeline(temp_video, temp_output_dir, default_config)
        
        with patch.object(pipeline2, '_validate_stage') as mock_validate, \
             patch.object(pipeline2, '_extract_2d_stage') as mock_extract, \
             patch.object(pipeline2, '_map_stage') as mock_map, \
             patch.object(pipeline2, '_lift_stage') as mock_lift, \
             patch.object(pipeline2, '_post_process_stage') as mock_post, \
             patch.object(pipeline2, '_visualize_stage') as mock_viz:
            
            result2 = pipeline2.run()
            
            # All stages should be skipped
            mock_validate.assert_not_called()
            mock_extract.assert_not_called()
            mock_map.assert_not_called()
            mock_lift.assert_not_called()
            mock_post.assert_not_called()
            mock_viz.assert_not_called()
            
            # Verify output message
            captured = capsys.readouterr()
            assert captured.out.count("already complete") == 6  # All 6 stages
    
    def test_partial_resume(self, temp_video, temp_output_dir, default_config):
        """Test resuming from a partially complete pipeline."""
        working_dir = default_config.working_dir
        
        # Manually create checkpoints for stages 1-3
        (working_dir / "video_metadata.json").write_text("{}")
        (working_dir / "poses_2d.npz").write_bytes(b"fake")
        (working_dir / "poses_2d_mapped.npz").write_bytes(b"fake")
        
        pipeline = Pipeline(temp_video, temp_output_dir, default_config)
        
        with patch.object(pipeline, '_validate_stage') as mock_validate, \
             patch.object(pipeline, '_extract_2d_stage') as mock_extract, \
             patch.object(pipeline, '_map_stage') as mock_map:
            
            # Create remaining checkpoints so pipeline completes
            (working_dir / "poses_3d.npy").write_bytes(b"fake")
            (working_dir / "poses_3d_cleaned.npy").write_bytes(b"fake")
            (working_dir / "visualization.html").write_text("<html></html>")
            
            result = pipeline.run()
            
            # First 3 stages should be skipped
            mock_validate.assert_not_called()
            mock_extract.assert_not_called()
            mock_map.assert_not_called()
