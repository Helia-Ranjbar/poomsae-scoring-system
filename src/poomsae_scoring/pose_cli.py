from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poomsae_scoring.scoring.pose_estimation import (
    PoseConfig,
    PoseEstimationPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate MediaPipe poses for isolated Poomsae athlete videos.",
    )
    parser.add_argument(
        "players_root",
        type=Path,
        nargs="?",
        default=Path("data/processed/players"),
        help="Processed players root containing <player>/angles/<angle>/crop.mp4.",
    )
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    parser.add_argument(
        "--output-size",
        type=int,
        default=640,
        help="Square MediaPipe input/output size.",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        default=None,
        help="Optional camera angles to process, for example: --angles 45",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum seconds processed from each video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not write annotated pose videos.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Process videos even when all requested outputs already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = PoseConfig(
            model_complexity=args.model_complexity,
            minimum_detection_confidence=args.detection_confidence,
            minimum_tracking_confidence=args.tracking_confidence,
            keypoint_confidence=args.keypoint_confidence,
            output_size=args.output_size,
            save_annotated_video=not args.no_video,
            maximum_duration_seconds=args.max_duration,
        )
        pipeline = PoseEstimationPipeline(config)
        results = pipeline.process_dataset(
            args.players_root,
            angles=set(args.angles) if args.angles else None,
            skip_completed=not args.reprocess,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if not results:
        print("No pending isolated videos found.")
        return
    for result in results:
        print(
            f"{result.player_id} angle {result.angle}: "
            f"{result.frames_detected}/{result.frames_processed} poses -> "
            f"{result.output_directory}"
        )


if __name__ == "__main__":
    main()
