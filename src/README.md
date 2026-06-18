# Source Code Modules

This directory contains the core modules for the motion-tracking pipeline.

## pose_3d_lifter.py

3D pose lifting module that integrates with DiffPose to convert 2D keypoint sequences into 3D joint trajectories.

### Main Functions

- `lift_2d_to_3d(input_2d_path, output_3d_path, config)` - Lifts 2D keypoints to 3D coordinates
- `load_diffpose_model(config)` - Loads DiffPose model from checkpoint

### Configuration

Use the `LiftingConfig` dataclass to configure the lifting process:

```python
from pathlib import Path
from pose_3d_lifter import lift_2d_to_3d, LiftingConfig

config = LiftingConfig(
    checkpoint_path=Path("checkpoints/diffpose.pth"),
    config_path=Path("configs/diffpose_config.yml"),  # optional
    device='cpu',  # or 'cuda'
    batch_size=1
)

num_frames = lift_2d_to_3d(
    input_2d_path=Path("data/keypoints_2d.npy"),
    output_3d_path=Path("data/keypoints_3d.npy"),
    config=config
)
```

### Input/Output Format

- **Input**: NPY file with shape `(frames, 16, 2)` - 2D keypoints in DiffPose format
- **Output**: NPY file with shape `(frames, 16, 3)` - 3D joint coordinates [x, y, z]

### Testing

Run tests with pytest:

```bash
pytest tests/test_pose_3d_lifter.py -v
```
