from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from poomsae_scoring.config import PreprocessingConfig

VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    input_path: str
    output_directory: str
    source_width: int
    source_height: int
    source_fps: float
    source_frame_count: int
    source_duration_seconds: float
    output_width: int
    output_height: int
    target_fps: float
    frames_written: int


def discover_videos(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def letterbox_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    fill_value: int = 0,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full(
        (target_height, target_width, frame.shape[2]),
        fill_value,
        dtype=frame.dtype,
    )
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


class VideoPreprocessor:
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def process(self, input_path: Path, output_root: Path) -> PreprocessingResult:
        input_path = input_path.expanduser().resolve()
        output_root = output_root.expanduser().resolve()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if source_fps <= 0:
                raise ValueError(f"Video reports an invalid FPS: {input_path}")

            video_output = output_root / input_path.stem
            frames_output = video_output / "frames"
            frames_output.mkdir(parents=True, exist_ok=True)

            frames_written = self._extract_frames(
                capture=capture,
                source_fps=source_fps,
                source_frame_count=source_frame_count,
                frames_output=frames_output,
                video_name=input_path.name,
            )
        finally:
            capture.release()

        source_duration = source_frame_count / source_fps
        result = PreprocessingResult(
            input_path=str(input_path),
            output_directory=str(video_output),
            source_width=source_width,
            source_height=source_height,
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            source_duration_seconds=source_duration,
            output_width=self.config.width,
            output_height=self.config.height,
            target_fps=self.config.target_fps,
            frames_written=frames_written,
        )
        metadata_path = video_output / "metadata.json"
        metadata_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result

    def _extract_frames(
        self,
        capture: cv2.VideoCapture,
        source_fps: float,
        source_frame_count: int,
        frames_output: Path,
        video_name: str,
    ) -> int:
        sample_interval = source_fps / self.config.target_fps
        next_sample_index = 0.0
        frame_index = 0
        output_index = 0
        max_source_frames = source_frame_count
        if self.config.max_duration_seconds is not None:
            max_source_frames = min(
                max_source_frames,
                round(self.config.max_duration_seconds * source_fps),
            )

        progress = tqdm(
            total=max_source_frames,
            desc=f"Preprocessing {video_name}",
            unit="frame",
        )
        try:
            while frame_index < max_source_frames:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index + 1e-9 >= next_sample_index:
                    processed_frame = letterbox_frame(
                        frame,
                        target_width=self.config.width,
                        target_height=self.config.height,
                    )
                    frame_path = frames_output / f"frame_{output_index:06d}.jpg"
                    write_success = cv2.imwrite(
                        str(frame_path),
                        processed_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                    )
                    if not write_success:
                        raise OSError(f"Could not write frame: {frame_path}")
                    output_index += 1
                    next_sample_index += sample_interval

                frame_index += 1
                progress.update(1)
        finally:
            progress.close()

        return output_index

