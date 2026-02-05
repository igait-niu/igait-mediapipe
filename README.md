# iGait 3D Pose Estimation

MediaPipe Holistic-based 3D pose estimation pipeline for gait analysis. Processes video files to extract 3D body, hand, and face landmarks, with optional ML-ready data export.

## Features

- 3D landmark extraction using MediaPipe Holistic (no 2D-to-3D lifting required)
- 543 landmarks per frame: 33 body + 21 per hand + 468 face
- GPU-accelerated processing with optional CPU fallback
- HIPAA-compliant skeleton-only output mode (strips original video)
- Temporal pose smoothing
- Batch processing for multiple videos via PBS job scripts

## Requirements

- Python 3.9+
- OpenCV
- MediaPipe
- NumPy
- PyTorch (optional, for GPU acceleration)

### Install

```bash
pip install mediapipe opencv-python numpy
```

For GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic

```bash
python3.9 3DPoseEstimation.py input_video.mp4
```

Output is saved to `output/<input_name>_pose.mp4` by default.

### CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `input` | Input video file path (positional, required) | — |
| `-o`, `--output` | Output video file path | `output/<input>_pose.mp4` |
| `-s`, `--start` | Start time in seconds | `0` |
| `-d`, `--duration` | Duration to process in seconds | Full video |
| `--batch-size` | Frame batch size for processing | `16` |
| `--no-gpu` | Force CPU-only processing | GPU enabled |
| `--no-smooth` | Disable temporal pose smoothing | Smoothing enabled |
| `--save-data` | Extract and save landmark data for ML training | Disabled |
| `--label` | Label for this video (used with `--save-data`) | `"unlabeled"` |
| `--skeleton-only` | HIPAA-compliant mode: output skeleton overlay on black background, no original video content | Disabled |

### Examples

Process a specific time range:

```bash
python3.9 3DPoseEstimation.py input_video.mp4 -s 30 -d 60
```

Extract ML training data with a label:

```bash
python3.9 3DPoseEstimation.py input_video.mp4 --save-data --label "neurotypical"
```

HIPAA-compliant skeleton-only output:

```bash
python3.9 3DPoseEstimation.py input_video.mp4 --skeleton-only --save-data --label "neurodivergent"
```

Custom output path with CPU processing:

```bash
python3.9 3DPoseEstimation.py input_video.mp4 -o results/output.mp4 --no-gpu
```

## HPC Batch Processing (NIU)

The included `run.pbs` script processes all videos in an input directory on NIU's HPC cluster.

### File Naming Convention

Video filenames determine automatic labeling based on the prefix before the first underscore:

| Prefix | Label |
|---|---|
| `1xx` (e.g., `101_walk.MOV`) | `neurotypical` |
| `2xx` (e.g., `203_walk.MOV`) | `neurodivergent` |

Files with unrecognized prefixes are skipped.

### Running on HPC

1. Place input videos in the `input/` directory on the cluster.
2. Submit the job:

```bash
qsub run.pbs
```

The PBS script requests 1 GPU, 16 CPUs, and 64 GB of memory with a 15-minute walltime. Adjust these values in `run.pbs` as needed.

## Output

```
output/
├── <video_name>_pose.mp4          # Video with pose overlay
├── <video_name>_skeleton.mp4      # Skeleton-only video (if --skeleton-only)
└── training_data/                 # ML data (if --save-data)
    ├── <video_name>_landmarks.json
```

### Training Data Format

When `--save-data` is enabled, a JSON file is generated per video (`_landmarks.json`) containing:

- `video_filename` — source video path
- `total_frames` — number of processed frames
- `label` — the assigned label string
- `hipaa_compliant` — whether skeleton-only mode was used
- `landmarks_data` — per-frame array with 3D coordinates for pose (33), left hand (21), right hand (21), and face (468) landmarks

Each landmark provides 3 coordinates (x, y, z). Missing landmarks are stored as `null`.
