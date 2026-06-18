# motion-tracking

Offline 3D human pose estimation for swimming videos using a Python-based workflow.

## Objective
Extract accurate 3D skeletal motion from single-camera swimming/underwater footage without real-time constraints.

## Prerequisites & Environment Setup
Use Python 3.10+ and create an isolated environment:

```bash
python -m venv swim_pose_env
source swim_pose_env/bin/activate  # Windows: swim_pose_env\Scripts\activate
pip install numpy opencv-python ultralytics scipy matplotlib
```

## Stage 1: High-Accuracy 2D Keypoint Extraction
This repository includes `extract_2d_joints.py` for offline per-frame 2D extraction using YOLOv8x-pose.

```bash
python /home/runner/work/motion-tracking/motion-tracking/extract_2d_joints.py \
  /absolute/path/to/your_swimming_video.mp4 \
  --output /absolute/path/to/swimmer_2d_output.npz
```

The output NPZ file stores a `poses` array containing `[x, y, confidence]` triplets for 17 joints per frame.

## Stage 2: 3D Temporal Diffusion Lifting (DiffPose)
Use DiffPose to lift 2D joints to temporally consistent 3D trajectories:

1. Clone DiffPose and install dependencies:
   ```bash
   git clone https://github.com/zjdcts/DiffPose.git
   cd DiffPose
   pip install -r requirements.txt
   ```
2. Download the Human3.6M checkpoints and place them in `./checkpoints`.
3. Convert `swimmer_2d_output.npz` into DiffPose/Human3.6M-compatible joint input.
4. Run lifting:
   ```bash
   python main_diffpose_frame.py \
     --config human36m_diffpose_uvxyz_cpn.yml \
     --model_pose_path checkpoints/gcn_xyz_cpn.pth \
     --model_diff_path checkpoints/diffpose_uvxyz_cpn.pth
   ```

## Stage 3: Aquatic Post-Processing & Refraction Correction
This repository includes `clean_swimming_data.py` to smooth 3D trajectories and provide a hook for bone-length/refraction constraints.

```bash
python /home/runner/work/motion-tracking/motion-tracking/clean_swimming_data.py \
  /absolute/path/to/diffpose_output.npy \
  --output /absolute/path/to/swimmer_3d_cleaned.npy
```

## Next Steps
- Add a helper script for YOLO-17 to DiffPose-16 keypoint mapping if needed.
- Add an integrated `matplotlib` 3D animation utility for side-by-side visualization.
