from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poomsae_scoring.preprocessing.isolation import (
    IsolationConfig,
    PlayerIsolationPipeline,
    discover_player_videos,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, track, and isolate each Poomsae player in every camera angle.",
    )
    parser.add_argument("input", type=Path, help="Video file or dataset directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/players"),
        help="Root directory organized by player and camera angle.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("notebooks/yolo26n-seg.pt"),
        help="Ultralytics instance-segmentation model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--recovery-confidence", type=float, default=0.15)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--recovery-image-size", type=int, default=960)
    parser.add_argument("--max-tracking-gap", type=int, default=8)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum seconds processed from each video.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        videos = discover_player_videos(args.input)
        if not videos:
            print(f"No supported videos found under {args.input}")
            return
        config = IsolationConfig(
            model_path=args.model.expanduser().resolve(),
            device=args.device,
            confidence=args.confidence,
            recovery_confidence=args.recovery_confidence,
            output_size=args.output_size,
            recovery_image_size=args.recovery_image_size,
            maximum_tracking_gap=args.max_tracking_gap,
            maximum_duration_seconds=args.max_duration,
        )
        pipeline = PlayerIsolationPipeline(config)
        results = pipeline.process_dataset(args.input, args.output)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for result in results:
        print(
            f"{result.player_id} angle {result.angle}: "
            f"{result.frames_isolated}/{result.frames_processed} frames -> "
            f"{result.output_directory}"
        )


if __name__ == "__main__":
    main()
