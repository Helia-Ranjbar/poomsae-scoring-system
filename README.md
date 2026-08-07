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
