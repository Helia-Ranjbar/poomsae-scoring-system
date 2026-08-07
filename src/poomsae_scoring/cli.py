from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poomsae_scoring.config import PreprocessingConfig
from poomsae_scoring.preprocessing import VideoPreprocessor, discover_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Poomsae videos for pose estimation and scoring.",
    )
    parser.add_argument("input", type=Path, help="A video file or directory of videos.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory for extracted frames and metadata.",
    )
    parser.add_argument("--width", type=int, default=640, help="Output frame width.")
    parser.add_argument("--height", type=int, default=640, help="Output frame height.")
    parser.add_argument("--fps", type=float, default=15.0, help="Sampling frame rate.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum number of seconds to process per video.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = PreprocessingConfig(
            width=args.width,
            height=args.height,
            target_fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            max_duration_seconds=args.max_duration,
        )
        videos = discover_videos(args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    if not videos:
        print(f"No supported videos found under {args.input}")
        return

    preprocessor = VideoPreprocessor(config)
    failures = 0
    for video in videos:
        try:
            result = preprocessor.process(video, args.output)
            print(
                f"Processed {video.name}: {result.frames_written} frames -> "
                f"{result.output_directory}"
            )
        except (OSError, ValueError) as error:
            failures += 1
            print(f"Failed {video}: {error}", file=sys.stderr)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

