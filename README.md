# iGait 3D Pose Estimation

MediaPipe Holistic-based pose estimation pipeline for gait analysis. Processes video files to extract body landmarks normalized to the video frame (0 to 1), with optional hand and face landmark processing and ML-ready data export.

## Features

- Landmark extraction using MediaPipe Holistic with coordinates normalized to the video frame (0 to 1)
- 33 body pose landmarks by default; optional hand (21 per hand) and face (468) landmarks
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

Output is saved to the `output/` directory by default.

### CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `input` | Input video file path (positional, required) | — |
| `-o`, `--output` | Output directory path | `./output` |
| `-s`, `--start` | Start time in seconds | `0` |
| `-d`, `--duration` | Duration to process in seconds | Full video |
| `--batch-size` | Frame batch size for processing | `16` |
| `--no-gpu` | Force CPU-only processing | GPU enabled |
| `--no-smooth` | Disable temporal pose smoothing | Smoothing enabled |
| `--save-data` | Extract and save landmark data for ML training | Disabled |
| `--label` | Label for this video (used with `--save-data`) | `"unlabeled"` |
| `--skeleton-only` | HIPAA-compliant mode: output skeleton overlay on black background, no original video content | Disabled |
| `--enable-hands` | Enable hand landmark processing (21 landmarks per hand) | Disabled |
| `--enable-face` | Enable face landmark processing (468 landmarks) | Disabled |

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

Enable hand and face landmarks:

```bash
python3.9 3DPoseEstimation.py input_video.mp4 --enable-hands --enable-face
```

Custom output directory with CPU processing:

```bash
python3.9 3DPoseEstimation.py input_video.mp4 -o results/ --no-gpu
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
<output_dir>/                          # Default: ./output, or -o <path>
├── <video_name>_pose.mp4             # Video with pose overlay
├── <video_name>_skeleton.mp4         # Skeleton-only video (if --skeleton-only)
└── training_data/                    # ML data (if --save-data)
    └── <video_name>_landmarks.json
```

### Training Data Format

When `--save-data` is enabled, a JSON file is generated per video (`_landmarks.json`) containing:

- `video_filename` — source video path
- `width` — video width in pixels
- `height` — video height in pixels
- `fps` — video frames per second
- `total_frames` — number of processed frames
- `label` — the assigned label string
- `hipaa_compliant` — whether skeleton-only mode was used
- `landmarks_data` — per-frame array with `frame_number` (0-based), `timestamp_sec` (video-relative time in seconds), and coordinates for pose (33) landmarks; left hand (21), right hand (21), and face (468) landmarks are included only when enabled via `--enable-hands` / `--enable-face`

Landmark coordinates use x and y normalized to the video frame (0 to 1), and z representing relative depth. Disabled or undetected landmarks are stored as `null`.

| Landmark type | Format | Notes |
|---|---|---|
| Pose (33) | `[x, y, z, visibility]` | `visibility` is a 0–1 confidence score from MediaPipe |
| Hand (21 per hand) | `[x, y, z]` | No confidence score available from MediaPipe |
| Face (468) | `[x, y, z]` | No confidence score available from MediaPipe |
