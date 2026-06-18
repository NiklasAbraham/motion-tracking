# 006 – Pipeline Orchestration

## Purpose

Provide a single entry point that chains all stages from input video to 3D visualization.

## Requirements

1. Accept a video file path and produce the final 3D visualization.
2. Run stages sequentially (each depends on the previous output).
3. Store intermediate artifacts in a configurable working directory.
4. Provide progress reporting (stage name, percentage).
5. Support resuming from intermediate artifacts if a stage has already completed.

## Interface

```python
def run_pipeline(video_path: Path, output_dir: Path, config: PipelineConfig) -> Path:
    """Run the full pipeline and return path to the visualization output."""
```

## Stage Order

1. Input validation
2. 2D pose estimation
3. Keypoint format mapping (YOLO → DiffPose)
4. 3D lifting
5. Post-processing
6. Visualization

## Configuration

```python
@dataclass
class PipelineConfig:
    model_2d: str = "yolov8x-pose.pt"
    lifting_config: str = "human36m_diffpose_uvxyz_cpn.yml"
    filter_cutoff: float = 3.0
    filter_fs: float = 30.0
    filter_order: int = 2
    viz_format: str = "html"
    working_dir: Path = Path("./pipeline_output")
```
