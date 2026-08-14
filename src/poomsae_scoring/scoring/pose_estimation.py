from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from poomsae_scoring.scoring.pose_features import (
    COCO_KEYPOINT_NAMES,
    extract_pose_features,
    pose_motion_energy,
)

MEDIAPIPE_POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
MEDIAPIPE_LANDMARK_INDEX = {
    name: index for index, name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES)
}
MEDIAPIPE_TO_COCO = {
    "nose": "nose",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
}
POSE_FEATURE_NAMES = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "support_knee_angle",
    "kicking_knee_angle",
    "support_side",
    "torso_lean_degrees",
    "shoulder_tilt_degrees",
    "hip_tilt_degrees",
    "stance_width_shoulders",
    "balance_offset_shoulders",
    "left_wrist_centerline",
    "right_wrist_centerline",
    "left_ankle_centerline",
    "right_ankle_centerline",
    "head_centerline",
    "mean_keypoint_confidence",
)


@dataclass(frozen=True, slots=True)
class PoseVideo:
    player_id: str
    angle: str
    path: Path

    @property
    def output_directory(self) -> Path:
        return self.path.parent / "pose"


@dataclass(frozen=True, slots=True)
class PoseConfig:
    model_complexity: int = 2
    minimum_detection_confidence: float = 0.5
    minimum_tracking_confidence: float = 0.5
    keypoint_confidence: float = 0.25
    output_size: int | None = 640
    smooth_landmarks: bool = True
    codec: str = "mp4v"
    save_annotated_video: bool = True
    maximum_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.model_complexity not in {0, 1, 2}:
            raise ValueError("MediaPipe model complexity must be 0, 1, or 2.")
        if not 0 <= self.minimum_detection_confidence <= 1:
            raise ValueError("Minimum detection confidence must be between 0 and 1.")
        if not 0 <= self.minimum_tracking_confidence <= 1:
            raise ValueError("Minimum tracking confidence must be between 0 and 1.")
        if not 0 <= self.keypoint_confidence <= 1:
            raise ValueError("Keypoint confidence must be between 0 and 1.")
        if self.output_size is not None and self.output_size <= 0:
            raise ValueError("Output size must be positive.")
        if len(self.codec) != 4:
            raise ValueError("Video codec must contain exactly four characters.")
        if self.maximum_duration_seconds is not None and self.maximum_duration_seconds <= 0:
            raise ValueError("Maximum duration must be positive.")


@dataclass(frozen=True, slots=True)
class PoseResult:
    player_id: str
    angle: str
    input_path: str
    output_directory: str
    width: int
    height: int
    fps: float
    frames_processed: int
    frames_detected: int
    detection_rate: float
    backend: str


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def discover_pose_videos(
    players_root: Path,
    angles: set[str] | None = None,
) -> list[PoseVideo]:
    root = players_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Processed players directory does not exist: {root}")

    videos: list[PoseVideo] = []
    for path in root.glob("*/angles/*/crop.mp4"):
        angle = path.parent.name
        if angles is not None and angle not in angles:
            continue
        videos.append(
            PoseVideo(
                player_id=path.parents[2].name,
                angle=angle,
                path=path.resolve(),
            )
        )
    videos.sort(key=lambda item: (_natural_key(item.player_id), float(item.angle)))
    return videos


def pose_output_is_complete(video: PoseVideo, save_annotated_video: bool = True) -> bool:
    required = ["keypoints.csv", "features.csv", "metadata.json"]
    if save_annotated_video:
        required.append("annotated.mp4")
    if not all((video.output_directory / filename).is_file() for filename in required):
        return False
    try:
        with (video.output_directory / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (json.JSONDecodeError, OSError):
        return False
    return (
        metadata.get("completed_full_video") is True
        and metadata.get("backend") == "MediaPipe Pose"
    )


def _load_mediapipe() -> ModuleType:
    try:
        import mediapipe
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe is required for pose estimation. Install the project dependencies."
        ) from error
    return mediapipe


def create_pose_estimator(config: PoseConfig, mediapipe_module: ModuleType | None = None) -> Any:
    mp = mediapipe_module if mediapipe_module is not None else _load_mediapipe()
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=config.model_complexity,
        smooth_landmarks=config.smooth_landmarks,
        min_detection_confidence=config.minimum_detection_confidence,
        min_tracking_confidence=config.minimum_tracking_confidence,
    )


def mediapipe_landmarks_to_arrays(
    landmarks: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(landmarks) != len(MEDIAPIPE_POSE_LANDMARK_NAMES):
        raise ValueError("Expected 33 MediaPipe pose landmarks.")
    points = np.array(
        [[float(landmark.x) * width, float(landmark.y) * height] for landmark in landmarks],
        dtype=float,
    )
    visibility = np.array(
        [float(landmark.visibility) for landmark in landmarks],
        dtype=float,
    )
    z_coordinates = np.array(
        [float(landmark.z) * width for landmark in landmarks],
        dtype=float,
    )
    return points, visibility, z_coordinates


def mediapipe_to_coco(
    keypoints: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    media_points = np.asarray(keypoints, dtype=float)
    media_confidence = np.asarray(confidence, dtype=float)
    if media_points.shape != (33, 2):
        raise ValueError("Expected MediaPipe keypoints with shape (33, 2).")
    if media_confidence.shape != (33,):
        raise ValueError("Expected MediaPipe confidence with shape (33,).")

    coco_points = np.empty((17, 2), dtype=float)
    coco_confidence = np.empty(17, dtype=float)
    coco_index = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}
    for media_name, coco_name in MEDIAPIPE_TO_COCO.items():
        source_index = MEDIAPIPE_LANDMARK_INDEX[media_name]
        target_index = coco_index[coco_name]
        coco_points[target_index] = media_points[source_index]
        coco_confidence[target_index] = media_confidence[source_index]
    return coco_points, coco_confidence


def draw_mediapipe_pose(
    frame: np.ndarray,
    pose_landmarks: Any,
    mediapipe_module: ModuleType | None = None,
) -> np.ndarray:
    mp = mediapipe_module if mediapipe_module is not None else _load_mediapipe()
    annotated = frame.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=(
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        ),
    )
    return annotated


def _keypoint_fieldnames() -> list[str]:
    fields = ["frame_index", "time_seconds", "status"]
    for name in MEDIAPIPE_POSE_LANDMARK_NAMES:
        fields.extend(
            (
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_confidence",
            )
        )
    return fields


def _keypoint_row(
    frame_index: int,
    fps: float,
    keypoints: np.ndarray | None,
    z_coordinates: np.ndarray | None,
    confidence: np.ndarray | None,
) -> dict[str, float | int | str]:
    detected = keypoints is not None and confidence is not None and z_coordinates is not None
    row: dict[str, float | int | str] = {
        "frame_index": frame_index,
        "time_seconds": frame_index / fps,
        "status": "detected" if detected else "missing",
    }
    for index, name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES):
        if detected:
            row[f"{name}_x"] = float(keypoints[index, 0])
            row[f"{name}_y"] = float(keypoints[index, 1])
            row[f"{name}_z"] = float(z_coordinates[index])
            row[f"{name}_confidence"] = float(confidence[index])
        else:
            row[f"{name}_x"] = math.nan
            row[f"{name}_y"] = math.nan
            row[f"{name}_z"] = math.nan
            row[f"{name}_confidence"] = math.nan
    return row


def _empty_pose_features() -> dict[str, float | str]:
    features: dict[str, float | str] = {
        name: math.nan for name in POSE_FEATURE_NAMES
    }
    features["support_side"] = "unknown"
    return features


class PoseEstimationPipeline:
    def __init__(
        self,
        config: PoseConfig,
        mediapipe_module: ModuleType | None = None,
    ) -> None:
        self.config = config
        self.mediapipe = (
            mediapipe_module if mediapipe_module is not None else _load_mediapipe()
        )

    def process_video(self, video: PoseVideo) -> PoseResult:
        capture = cv2.VideoCapture(str(video.path))
        if not capture.isOpened():
            raise OSError(f"Could not open pose input video: {video.path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or source_width <= 0 or source_height <= 0:
            capture.release()
            raise ValueError(f"Invalid video metadata: {video.path}")

        if self.config.output_size is None:
            width, height = source_width, source_height
        else:
            width = height = self.config.output_size

        frame_limit = source_frames
        if self.config.maximum_duration_seconds is not None:
            frame_limit = min(
                source_frames,
                max(1, round(self.config.maximum_duration_seconds * fps)),
            )

        output_directory = video.output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        keypoints_path = output_directory / "keypoints.csv"
        features_path = output_directory / "features.csv"
        annotated_path = output_directory / "annotated.mp4"

        writer = None
        if self.config.save_annotated_video:
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*self.config.codec),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                capture.release()
                raise OSError(f"Could not create annotated video: {annotated_path}")

        frames_processed = 0
        frames_detected = 0
        previous_keypoints: np.ndarray | None = None
        previous_confidence: np.ndarray | None = None
        feature_fields = [
            "frame_index",
            "time_seconds",
            "status",
            "motion_energy",
            *POSE_FEATURE_NAMES,
        ]

        try:
            with (
                keypoints_path.open("w", newline="", encoding="utf-8") as keypoint_file,
                features_path.open("w", newline="", encoding="utf-8") as feature_file,
                create_pose_estimator(self.config, self.mediapipe) as pose,
            ):
                keypoint_writer = csv.DictWriter(
                    keypoint_file,
                    fieldnames=_keypoint_fieldnames(),
                )
                keypoint_writer.writeheader()
                feature_writer = csv.DictWriter(
                    feature_file,
                    fieldnames=feature_fields,
                )
                feature_writer.writeheader()

                progress = tqdm(
                    total=frame_limit,
                    desc=f"MediaPipe pose {video.player_id}-{video.angle}",
                    unit="frame",
                )
                while frames_processed < frame_limit:
                    success, frame = capture.read()
                    if not success:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))

                    frame_index = frames_processed
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(image_rgb)
                    if result.pose_landmarks is None:
                        keypoints = None
                        z_coordinates = None
                        confidence = None
                        features = _empty_pose_features()
                        motion_energy = math.nan
                        annotated = frame
                        previous_keypoints = None
                        previous_confidence = None
                    else:
                        frames_detected += 1
                        keypoints, confidence, z_coordinates = (
                            mediapipe_landmarks_to_arrays(
                                result.pose_landmarks.landmark,
                                width,
                                height,
                            )
                        )
                        coco_keypoints, coco_confidence = mediapipe_to_coco(
                            keypoints,
                            confidence,
                        )
                        features = extract_pose_features(
                            coco_keypoints,
                            coco_confidence,
                            minimum_confidence=self.config.keypoint_confidence,
                        )
                        motion_energy = (
                            pose_motion_energy(
                                previous_keypoints,
                                coco_keypoints,
                                previous_confidence,
                                coco_confidence,
                                fps,
                                minimum_confidence=self.config.keypoint_confidence,
                            )
                            if previous_keypoints is not None
                            and previous_confidence is not None
                            else math.nan
                        )
                        annotated = draw_mediapipe_pose(
                            frame,
                            result.pose_landmarks,
                            self.mediapipe,
                        )
                        previous_keypoints = coco_keypoints
                        previous_confidence = coco_confidence

                    keypoint_writer.writerow(
                        _keypoint_row(
                            frame_index,
                            fps,
                            keypoints,
                            z_coordinates,
                            confidence,
                        )
                    )
                    feature_writer.writerow(
                        {
                            "frame_index": frame_index,
                            "time_seconds": frame_index / fps,
                            "status": (
                                "detected"
                                if result.pose_landmarks is not None
                                else "missing"
                            ),
                            "motion_energy": motion_energy,
                            **features,
                        }
                    )
                    if writer is not None:
                        writer.write(annotated)
                    frames_processed += 1
                    progress.update(1)
                progress.close()
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        detection_rate = (
            frames_detected / frames_processed if frames_processed else 0.0
        )
        result = PoseResult(
            player_id=video.player_id,
            angle=video.angle,
            input_path=str(video.path),
            output_directory=str(output_directory),
            width=width,
            height=height,
            fps=fps,
            frames_processed=frames_processed,
            frames_detected=frames_detected,
            detection_rate=detection_rate,
            backend="MediaPipe Pose",
        )
        metadata = {
            **asdict(result),
            "source_frames": source_frames,
            "completed_full_video": (
                frames_processed >= source_frames
                and self.config.maximum_duration_seconds is None
            ),
            "model_complexity": self.config.model_complexity,
            "minimum_detection_confidence": (
                self.config.minimum_detection_confidence
            ),
            "minimum_tracking_confidence": (
                self.config.minimum_tracking_confidence
            ),
            "keypoint_confidence": self.config.keypoint_confidence,
            "smooth_landmarks": self.config.smooth_landmarks,
            "maximum_duration_seconds": self.config.maximum_duration_seconds,
            "keypoint_format": "MediaPipe Pose 33",
            "keypoint_names": list(MEDIAPIPE_POSE_LANDMARK_NAMES),
        }
        with (output_directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        return result

    def process_dataset(
        self,
        players_root: Path,
        angles: set[str] | None = None,
        skip_completed: bool = True,
    ) -> list[PoseResult]:
        videos = discover_pose_videos(players_root, angles)
        if skip_completed:
            videos = [
                video
                for video in videos
                if not pose_output_is_complete(
                    video,
                    save_annotated_video=self.config.save_annotated_video,
                )
            ]
        return [self.process_video(video) for video in videos]
