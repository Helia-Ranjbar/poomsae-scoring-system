# Poomsae Taekwondo Scoring System

Starter project for building an automated Poomsae scoring system. The current
foundation focuses on preparing videos for later pose estimation, technique
segmentation, and scoring models.

## Current preprocessing features

- Accepts one video or a directory of videos.
- Samples videos at a configurable frame rate.
- Resizes frames with aspect-ratio-preserving letterboxing.
- Writes consistently numbered JPEG frames.
- Records source and processing metadata as JSON.
- Supports a maximum duration for long recordings.
- Skips unreadable videos without stopping a directory batch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Usage

Place videos under `data/raw`, then run:

```bash
poomsae-preprocess data/raw --output data/processed
```

Example with custom settings:

```bash
poomsae-preprocess data/raw \
  --output data/processed \
  --width 640 \
  --height 640 \
  --fps 15 \
  --max-duration 120
```

Each input video produces:

```text
data/processed/<video-name>/
├── frames/
│   ├── frame_000000.jpg
│   └── ...
└── metadata.json
```

## Front-athlete isolation notebook

Open `notebooks/front_athlete_preprocessing.ipynb` to detect the center/front
athlete, preserve the athlete's identity with ByteTrack, remove the two
background athletes, and create a normalized crop for later pose scoring.

The notebook is configured for the current video in `data/raw`. It first shows
numbered person candidates so the automatically selected athlete can be
confirmed or manually changed before the full video is processed.

## Dataset player-isolation pipeline

The reusable pipeline groups videos by the `<player>-<angle>` filename pattern,
finds the athlete standing nearest the center of the competition mat, tracks
that identity through the full recording with ByteTrack, and isolates the
athlete with YOLO instance-segmentation masks.

For the current dataset, files such as `WP12-0.MP4`, `WP12-45.MP4`, and
`WP12-135.MP4` are treated as three camera views of player `WP12`.

```bash
poomsae-isolate data/raw \
  --output data/processed/players \
  --model notebooks/yolo26n-seg.pt \
  --device cuda:0
```

Use `--device auto` to select CUDA when available and otherwise use the CPU.
Use `--max-duration 10` for a short end-to-end trial before processing the
complete dataset.

The processed dataset is organized by player first and angle second:

```text
data/processed/players/
├── manifest.json
└── WP12/
    ├── manifest.json
    └── angles/
        ├── 0/
        │   ├── isolated.mp4
        │   ├── crop.mp4
        │   ├── tracking.csv
        │   └── metadata.json
        ├── 45/
        └── 135/
```

- `isolated.mp4` preserves the source resolution and blacks out everything
  except the selected athlete.
- `crop.mp4` centers the isolated athlete on a square canvas for pose models.
- `tracking.csv` records frame-level track IDs, boxes, confidence, and missing
  or recovered frames.
- Manifests make all available player-angle pairs discoverable by later scoring
  stages.

### GPU inference

The notebook requires `cuda:0` by default. All YOLO segmentation, tracking, and
pose inference calls use that GPU, so the workload cannot silently fall back to
the CPU.

Before starting Jupyter, verify that the NVIDIA driver and CUDA-enabled PyTorch
build are available in the same virtual environment:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Install the CUDA-enabled PyTorch build appropriate for the installed NVIDIA
driver, then reinstall the project dependencies if needed. The GPU setting is:

```python
INFERENCE_DEVICE = "cuda:0"
```

Use `"cuda:1"` for the second GPU. Use `"auto"` only when CPU fallback is
acceptable. The notebook raises a clear error at startup if CUDA is unavailable.

## Project layout

```text
src/poomsae_scoring/
├── cli.py
├── config.py
└── preprocessing/
    └── video.py
tests/
data/
├── raw/
└── processed/
```

## Suggested next stages

1. Athlete detection and tracking.
2. 2D/3D pose landmark extraction.
3. Poomsae movement segmentation.
4. Stance, balance, trajectory, timing, and power features.
5. Rule-based and learned scoring calibrated to judge annotations.
