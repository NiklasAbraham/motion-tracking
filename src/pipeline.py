"""Pipeline orchestrator for end-to-end video-to-3D motion tracking.

This module chains all stages together and provides a CLI entry point.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline run.
    
    Attributes:
        model_2d: Path to YOLOv8x-pose model file
        lifting_config: Path to DiffPose lifting configuration
        filter_cutoff: Low-pass filter cutoff frequency in Hz
        filter_fs: Video sampling rate in Hz
        filter_order: Butterworth filter order
        viz_format: Visualization output format (html, mp4, or matplotlib)
        working_dir: Directory for intermediate artifacts
    """
    model_2d: str = "yolov8x-pose.pt"
    lifting_config: str = "human36m_diffpose_uvxyz_cpn.yml"
    filter_cutoff: float = 3.0
    filter_fs: float = 30.0
    filter_order: int = 2
    viz_format: str = "html"
    working_dir: Path = Path("./pipeline_output")
    
    def __post_init__(self):
        """Ensure working_dir is a Path object."""
        if not isinstance(self.working_dir, Path):
            self.working_dir = Path(self.working_dir)


class PipelineStage:
    """Represents a single pipeline stage with checkpoint tracking."""
    
    def __init__(self, name: str, checkpoint_file: str):
        """Initialize a pipeline stage.
        
        Args:
            name: Human-readable name of the stage
            checkpoint_file: Name of the output file that marks completion
        """
        self.name = name
        self.checkpoint_file = checkpoint_file
    
    def is_complete(self, working_dir: Path) -> bool:
        """Check if this stage has already been completed.
        
        Args:
            working_dir: Directory containing intermediate artifacts
            
        Returns:
            True if the stage output file exists, False otherwise
        """
        checkpoint_path = working_dir / self.checkpoint_file
        return checkpoint_path.exists()


class Pipeline:
    """Orchestrates the full video-to-3D pipeline."""
    
    # Define all pipeline stages in order
    STAGES = [
        PipelineStage("validate", "video_metadata.json"),
        PipelineStage("2d_extract", "poses_2d.npz"),
        PipelineStage("map", "poses_2d_mapped.npz"),
        PipelineStage("lift", "poses_3d.npy"),
        PipelineStage("post_process", "poses_3d_cleaned.npy"),
        PipelineStage("visualize", "visualization.html"),
    ]
    
    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ):
        """Initialize the pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Path to output directory for final results
            config: Pipeline configuration
            progress_callback: Optional callback(stage_name, percentage) for progress updates
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.config = config
        self.progress_callback = progress_callback
        
        # Ensure directories exist
        self.config.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _report_progress(self, stage_name: str, percentage: int):
        """Report progress to callback if available.
        
        Args:
            stage_name: Name of the current stage
            percentage: Progress percentage (0-100)
        """
        if self.progress_callback:
            self.progress_callback(stage_name, percentage)
    
    def _validate_stage(self) -> Path:
        """Stage 1: Validate input video and extract metadata.
        
        Returns:
            Path to the metadata JSON file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video is invalid
        """
        self._report_progress("validate", 0)
        
        # Check video exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Placeholder: In a real implementation, this would call validate_video()
        # from the input validator module and extract actual metadata
        metadata = {
            "path": str(self.video_path),
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 100,
            "duration_seconds": 3.33,
            "codec": "h264"
        }
        
        output_path = self.config.working_dir / "video_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._report_progress("validate", 100)
        return output_path
    
    def _extract_2d_stage(self) -> Path:
        """Stage 2: Extract 2D keypoints from video.
        
        Returns:
            Path to the 2D poses NPZ file
        """
        self._report_progress("2d_extract", 0)
        
        # Placeholder: In a real implementation, this would call extract_2d_keypoints()
        # from the 2D pose estimation module
        import numpy as np
        
        # Create dummy output
        output_path = self.config.working_dir / "poses_2d.npz"
        dummy_poses = np.random.rand(100, 17, 3)  # 100 frames, 17 joints, (x, y, conf)
        np.savez(output_path, poses=dummy_poses)
        
        self._report_progress("2d_extract", 100)
        return output_path
    
    def _map_stage(self) -> Path:
        """Stage 3: Map YOLO 17-joint to DiffPose 16-joint format.
        
        Returns:
            Path to the mapped poses NPZ file
        """
        self._report_progress("map", 0)
        
        # Placeholder: In a real implementation, this would call map_yolo_to_diffpose()
        # from the keypoint mapper module
        import numpy as np
        
        # Load input
        input_path = self.config.working_dir / "poses_2d.npz"
        poses_2d = np.load(input_path)['poses']
        
        # Create dummy mapped output (16 joints instead of 17)
        output_path = self.config.working_dir / "poses_2d_mapped.npz"
        mapped_poses = poses_2d[:, :16, :2]  # Drop last joint and confidence
        np.savez(output_path, poses=mapped_poses)
        
        self._report_progress("map", 100)
        return output_path
    
    def _lift_stage(self) -> Path:
        """Stage 4: Lift 2D poses to 3D.
        
        Returns:
            Path to the 3D poses NPY file
        """
        self._report_progress("lift", 0)
        
        # Placeholder: In a real implementation, this would call lift_2d_to_3d()
        # from the 3D lifting module
        import numpy as np
        
        # Load input
        input_path = self.config.working_dir / "poses_2d_mapped.npz"
        poses_2d = np.load(input_path)['poses']
        
        # Create dummy 3D output
        output_path = self.config.working_dir / "poses_3d.npy"
        poses_3d = np.random.rand(poses_2d.shape[0], 16, 3)  # Add z dimension
        np.save(output_path, poses_3d)
        
        self._report_progress("lift", 100)
        return output_path
    
    def _post_process_stage(self) -> Path:
        """Stage 5: Apply post-processing and smoothing.
        
        Returns:
            Path to the cleaned 3D poses NPY file
        """
        self._report_progress("post_process", 0)
        
        # Placeholder: In a real implementation, this would call clean_swimming_data()
        # from the post-processing module
        import numpy as np
        
        # Load input
        input_path = self.config.working_dir / "poses_3d.npy"
        poses_3d = np.load(input_path)
        
        # Create dummy cleaned output (just copy for now)
        output_path = self.config.working_dir / "poses_3d_cleaned.npy"
        np.save(output_path, poses_3d)
        
        self._report_progress("post_process", 100)
        return output_path
    
    def _visualize_stage(self) -> Path:
        """Stage 6: Generate 3D visualization.
        
        Returns:
            Path to the visualization output file
        """
        self._report_progress("visualize", 0)
        
        # Placeholder: In a real implementation, this would call visualize_3d()
        # from the visualization module
        output_filename = f"visualization.{self.config.viz_format}"
        output_path = self.config.working_dir / output_filename
        
        # Create dummy HTML visualization
        if self.config.viz_format == "html":
            with open(output_path, 'w') as f:
                f.write("<html><body><h1>3D Visualization Placeholder</h1></body></html>")
        
        self._report_progress("visualize", 100)
        return output_path
    
    def run(self) -> Path:
        """Run the full pipeline or resume from last checkpoint.
        
        Returns:
            Path to the final visualization output
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video is invalid or processing fails
        """
        # Map stages to their execution functions
        stage_functions = {
            "validate": self._validate_stage,
            "2d_extract": self._extract_2d_stage,
            "map": self._map_stage,
            "lift": self._lift_stage,
            "post_process": self._post_process_stage,
            "visualize": self._visualize_stage,
        }
        
        # Find the first incomplete stage
        start_stage_idx = 0
        for i, stage in enumerate(self.STAGES):
            if stage.is_complete(self.config.working_dir):
                print(f"✓ Stage '{stage.name}' already complete, skipping...")
                start_stage_idx = i + 1
            else:
                break
        
        # Run remaining stages
        for stage in self.STAGES[start_stage_idx:]:
            print(f"Running stage: {stage.name}")
            stage_func = stage_functions[stage.name]
            stage_func()
        
        # Copy final visualization to output directory
        viz_filename = f"visualization.{self.config.viz_format}"
        final_output = self.output_dir / viz_filename
        
        source = self.config.working_dir / viz_filename
        if source.exists():
            import shutil
            shutil.copy2(source, final_output)
        
        print(f"✓ Pipeline complete! Output: {final_output}")
        return final_output


def run_pipeline(video_path: Path, output_dir: Path, config: PipelineConfig) -> Path:
    """Run the full pipeline and return path to the visualization output.
    
    This is the main entry point for the pipeline orchestration.
    
    Args:
        video_path: Path to the input video file
        output_dir: Path to the output directory for final results
        config: Pipeline configuration settings
        
    Returns:
        Path to the final visualization output file
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video is invalid or any stage fails
    """
    pipeline = Pipeline(video_path, output_dir, config)
    return pipeline.run()


def main():
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Motion tracking pipeline for swimming videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m src.pipeline video.mp4 --output ./results
  python -m src.pipeline video.mp4 --output ./results --filter-cutoff 5.0
        """
    )
    
    parser.add_argument(
        "video",
        type=Path,
        help="Path to input video file (MP4)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory for final results (default: ./output)"
    )
    
    parser.add_argument(
        "--model-2d",
        type=str,
        default="yolov8x-pose.pt",
        help="YOLOv8x-pose model file (default: yolov8x-pose.pt)"
    )
    
    parser.add_argument(
        "--lifting-config",
        type=str,
        default="human36m_diffpose_uvxyz_cpn.yml",
        help="DiffPose configuration file (default: human36m_diffpose_uvxyz_cpn.yml)"
    )
    
    parser.add_argument(
        "--filter-cutoff",
        type=float,
        default=3.0,
        help="Low-pass filter cutoff frequency in Hz (default: 3.0)"
    )
    
    parser.add_argument(
        "--filter-fs",
        type=float,
        default=30.0,
        help="Video sampling rate in Hz (default: 30.0)"
    )
    
    parser.add_argument(
        "--filter-order",
        type=int,
        default=2,
        help="Butterworth filter order (default: 2)"
    )
    
    parser.add_argument(
        "--viz-format",
        type=str,
        choices=["html", "mp4", "matplotlib"],
        default="html",
        help="Visualization output format (default: html)"
    )
    
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("./pipeline_output"),
        help="Working directory for intermediate artifacts (default: ./pipeline_output)"
    )
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = PipelineConfig(
        model_2d=args.model_2d,
        lifting_config=args.lifting_config,
        filter_cutoff=args.filter_cutoff,
        filter_fs=args.filter_fs,
        filter_order=args.filter_order,
        viz_format=args.viz_format,
        working_dir=args.working_dir
    )
    
    try:
        output_path = run_pipeline(args.video, args.output, config)
        print(f"\n✓ Success! Visualization saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
