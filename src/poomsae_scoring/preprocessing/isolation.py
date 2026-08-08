from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from poomsae_scoring.device import resolve_inference_device
from poomsae_scoring.preprocessing.athlete import (
    box_bottom_contact_points,
    box_iou,
    interpolate_box_gaps,
    isolate_person,
    padded_crop,
    select_tracking_candidate,
)
from poomsae_scoring.preprocessing.video import VIDEO_EXTENSIONS, letterbox_frame

VIDEO_ID_PATTERN = re.compile(
    r"^(?P<player>.+?)[_-](?P<angle>-?\d+(?:\.\d+)?)$",
)

CALIBRATED_MAT_POLYGONS = {
    "0": np.array(
        [
            (0.60, 0.48),
            (0.95, 0.61),
            (0.95, 0.78),
            (0.90, 0.90),
            (0.71, 0.97),
            (0.25, 0.92),
            (0.05, 0.75),
            (0.05, 0.63),
        ],
        dtype=float,
    ),
    "45": np.array(
        [
            (0.15, 0.49),
            (0.33, 0.49),
            (0.86, 0.76),
            (0.94, 0.85),
            (0.72, 0.98),
            (0.12, 0.98),
            (0.04, 0.90),
            (0.04, 0.76),
        ],
        dtype=float,
    ),
    "135": np.array(
        [
            (0.09, 0.45),
            (0.70, 0.49),
            (0.86, 0.53),
            (0.99, 0.61),
            (0.84, 0.70),
            (0.25, 0.72),
            (0.10, 0.63),
            (0.05, 0.54),
        ],
        dtype=float,
    ),
}


@dataclass(frozen=True, slots=True)
class PlayerVideo:
    player_id: str
    angle: str
    path: Path


@dataclass(frozen=True, slots=True)
class IsolationConfig:
    model_path: Path = Path("notebooks/yolo26n-seg.pt")
    device: str = "auto"
    confidence: float = 0.35
    recovery_confidence: float = 0.15
    output_size: int = 640
    recovery_image_size: int = 960
    reference_position: float = 0.05
    maximum_tracking_gap: int = 8
    crop_padding: float = 0.25
    codec: str = "mp4v"
    maximum_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not 0 < self.recovery_confidence <= 1:
            raise ValueError("Recovery confidence must be between 0 and 1.")
        if self.output_size <= 0 or self.recovery_image_size <= 0:
            raise ValueError("Inference and output sizes must be positive.")
        if not 0 <= self.reference_position <= 1:
            raise ValueError("Reference position must be between 0 and 1.")
        if self.maximum_tracking_gap < 0:
            raise ValueError("Maximum tracking gap cannot be negative.")
        if self.crop_padding < 0:
            raise ValueError("Crop padding cannot be negative.")
        if len(self.codec) != 4:
            raise ValueError("Video codec must contain exactly four characters.")
        if self.maximum_duration_seconds is not None and self.maximum_duration_seconds <= 0:
            raise ValueError("Maximum duration must be positive.")


@dataclass(frozen=True, slots=True)
class IsolationResult:
    player_id: str
    angle: str
    input_path: str
    output_directory: str
    source_width: int
    source_height: int
    fps: float
    frames_processed: int
    frames_isolated: int
    frames_recovered: int
    reference_frame: int
    mat_detection: str
    device: str


def parse_player_video(path: Path) -> PlayerVideo:
    match = VIDEO_ID_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Video name must end with a player and angle such as WP1-45: {path.name}"
        )
    return PlayerVideo(
        player_id=match.group("player"),
        angle=_normalize_angle(match.group("angle")),
        path=path.expanduser().resolve(),
    )


def discover_player_videos(input_path: Path) -> list[PlayerVideo]:
    resolved = input_path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {resolved.suffix}")
        return [parse_player_video(resolved)]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")

    videos = [
        parse_player_video(path)
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda video: (_natural_key(video.player_id), _angle_sort_key(video.angle)))
    duplicates: dict[tuple[str, str], list[Path]] = {}
    for video in videos:
        duplicates.setdefault((video.player_id, video.angle), []).append(video.path)
    repeated = {key: paths for key, paths in duplicates.items() if len(paths) > 1}
    if repeated:
        descriptions = "; ".join(
            f"{player}/{angle}: {', '.join(str(path) for path in paths)}"
            for (player, angle), paths in repeated.items()
        )
        raise ValueError(f"Duplicate player-angle videos found: {descriptions}")
    return videos


def group_player_videos(videos: Iterable[PlayerVideo]) -> dict[str, list[PlayerVideo]]:
    grouped: dict[str, list[PlayerVideo]] = {}
    for video in videos:
        grouped.setdefault(video.player_id, []).append(video)
    for player_videos in grouped.values():
        player_videos.sort(key=lambda video: _angle_sort_key(video.angle))
    return dict(sorted(grouped.items(), key=lambda item: _natural_key(item[0])))


def player_angle_output_directory(
    output_root: Path,
    player_id: str,
    angle: str,
) -> Path:
    return output_root.expanduser().resolve() / player_id / "angles" / angle


def _normalize_angle(angle: str) -> str:
    value = float(angle)
    return str(int(value)) if value.is_integer() else format(value, "g")


def _angle_sort_key(angle: str) -> float:
    return float(angle)


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _load_yolo(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for athlete isolation. Install the project dependencies."
        ) from error
    return YOLO(str(model_path))


def _to_numpy(value: Any) -> np.ndarray:
    return value.cpu().numpy()


def _empty_detections() -> dict[str, np.ndarray]:
    return {
        "boxes": np.empty((0, 4), dtype=float),
        "ids": np.empty(0, dtype=int),
    }


def mask_ground_contact(
    mask: np.ndarray,
    box: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    resized_mask = np.asarray(mask)
    if resized_mask.shape != (image_height, image_width):
        resized_mask = cv2.resize(
            resized_mask.astype(np.float32),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
    y_coordinates, x_coordinates = np.nonzero(resized_mask > 0.5)
    if y_coordinates.size == 0:
        x1, _, x2, y2 = np.asarray(box, dtype=float)
        return np.array([(x1 + x2) / 2, y2], dtype=float)

    person_height = max(float(box[3] - box[1]), 1.0)
    bottom_y = float(np.percentile(y_coordinates, 99.5))
    bottom_band = y_coordinates >= bottom_y - max(3.0, 0.03 * person_height)
    return np.array(
        [
            float(np.median(x_coordinates[bottom_band])),
            float(np.median(y_coordinates[bottom_band])),
        ],
        dtype=float,
    )


def plausible_same_person(
    candidate_box: np.ndarray,
    previous_box: np.ndarray,
    frame_width: int,
    frame_height: int,
    maximum_center_step: float = 0.08,
    minimum_area_ratio: float = 0.35,
    maximum_area_ratio: float = 4.0,
) -> bool:
    candidate = np.asarray(candidate_box, dtype=float)
    previous = np.asarray(previous_box, dtype=float)
    candidate_center = np.array([(candidate[0] + candidate[2]) / 2, candidate[3]])
    previous_center = np.array([(previous[0] + previous[2]) / 2, previous[3]])
    normalized_step = np.linalg.norm(
        (candidate_center - previous_center) / np.array([frame_width, frame_height]),
    )
    candidate_area = max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
    previous_area = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
    area_ratio = candidate_area / previous_area
    return box_iou(candidate, previous) >= 0.05 or (
        normalized_step <= maximum_center_step
        and minimum_area_ratio <= area_ratio <= maximum_area_ratio
    )


def detect_competition_mat(frame: np.ndarray, angle: str) -> tuple[np.ndarray, str]:
    image_height, image_width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 80, 70], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )
    yellow_mask[: round(0.35 * image_height)] = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = 0.001 * image_width * image_height
    components = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= minimum_area
        and sum(cv2.boundingRect(contour)[1::2]) >= 0.45 * image_height
    ]
    automatic_polygon = _automatic_mat_polygon(components, image_width, image_height)
    if automatic_polygon is not None:
        return automatic_polygon, "automatic_yellow_border"

    calibration = CALIBRATED_MAT_POLYGONS.get(angle)
    if calibration is None:
        raise RuntimeError(
            f"Could not detect the competition mat and no calibration exists for angle {angle}."
        )
    polygon = np.rint(calibration * np.array([image_width, image_height])).astype(np.int32)
    return polygon, "camera_calibration"


def _automatic_mat_polygon(
    components: list[np.ndarray],
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    if not components:
        return None
    outer_hull = cv2.convexHull(np.concatenate(components, axis=0))
    perimeter = cv2.arcLength(outer_hull, True)
    frame_diagonal = math.hypot(image_width, image_height)
    candidates: list[tuple[float, np.ndarray]] = []
    for epsilon_ratio in np.linspace(0.0005, 0.03, 296):
        polygon = cv2.approxPolyDP(
            outer_hull,
            float(epsilon_ratio * perimeter),
            True,
        ).reshape(-1, 2)
        if (
            len(polygon) == 8
            and cv2.isContourConvex(polygon.reshape(-1, 1, 2))
            and np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1).min()
            >= 0.04 * frame_diagonal
            and cv2.contourArea(polygon) >= 0.10 * image_width * image_height
        ):
            candidates.append((epsilon_ratio, polygon))
    if not candidates:
        return None

    _, outer_polygon = max(candidates, key=lambda item: item[0])
    moments = cv2.moments(outer_polygon)
    if moments["m00"] == 0:
        return None
    center = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=float,
    )
    return np.rint(center + 0.90 * (outer_polygon - center)).astype(np.int32)


def select_mat_athlete(
    boxes: np.ndarray,
    masks: np.ndarray,
    mat_polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> int:
    if len(boxes) == 0:
        raise ValueError("No person candidates were provided.")
    moments = cv2.moments(mat_polygon)
    if moments["m00"] == 0:
        raise ValueError("Competition-mat polygon has zero area.")
    mat_center = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=float,
    )
    contacts = np.array(
        [
            mask_ground_contact(mask, box, frame_width, frame_height)
            for mask, box in zip(masks, boxes, strict=True)
        ],
    )
    on_mat = np.array(
        [
            cv2.pointPolygonTest(
                mat_polygon.astype(np.float32),
                tuple(contact.astype(float)),
                False,
            )
            >= 0
            for contact in contacts
        ],
        dtype=bool,
    )
    center_distances = np.linalg.norm(
        (contacts - mat_center) / np.array([frame_width, frame_height]),
        axis=1,
    )
    if np.any(on_mat):
        return int(np.argmin(np.where(on_mat, center_distances, np.inf)))

    boundary_distances = np.array(
        [
            abs(
                cv2.pointPolygonTest(
                    mat_polygon.astype(np.float32),
                    tuple(contact.astype(float)),
                    True,
                )
            )
            for contact in contacts
        ],
        dtype=float,
    )
    normalized_boundary_distances = boundary_distances / math.hypot(
        frame_width,
        frame_height,
    )
    fallback_scores = normalized_boundary_distances + 0.1 * center_distances
    return int(np.argmin(fallback_scores))


class PlayerIsolationPipeline:
    def __init__(
        self,
        config: IsolationConfig,
        model_factory: Callable[[Path], Any] = _load_yolo,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.device = resolve_inference_device(config.device)

    def process_dataset(
        self,
        input_path: Path,
        output_root: Path,
    ) -> list[IsolationResult]:
        videos = discover_player_videos(input_path)
        grouped = group_player_videos(videos)
        results: list[IsolationResult] = []
        for player_id, player_videos in grouped.items():
            player_results = [
                self.process_video(video, output_root)
                for video in player_videos
            ]
            results.extend(player_results)
            self._write_player_manifest(output_root, player_id, player_results)
        self._write_dataset_manifest(output_root, grouped, results)
        return results

    def process_video(
        self,
        video: PlayerVideo,
        output_root: Path,
    ) -> IsolationResult:
        capture = cv2.VideoCapture(str(video.path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video.path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if fps <= 0 or frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Video reports invalid metadata: {video.path}")

        if self.config.maximum_duration_seconds is not None:
            frame_count = min(
                frame_count,
                round(self.config.maximum_duration_seconds * fps),
            )
        reference_frame_index = min(
            frame_count - 1,
            round((frame_count - 1) * self.config.reference_position),
        )
        reference_frame = self._read_frame(video.path, reference_frame_index)
        mat_polygon, mat_detection = detect_competition_mat(reference_frame, video.angle)

        segmentation_model = self.model_factory(self.config.model_path)
        reference_result = segmentation_model.predict(
            reference_frame,
            device=self.device,
            classes=[0],
            conf=self.config.confidence,
            retina_masks=True,
            verbose=False,
        )[0]
        if (
            reference_result.boxes is None
            or len(reference_result.boxes) == 0
            or reference_result.masks is None
        ):
            raise RuntimeError(f"No segmented people found in reference frame: {video.path}")
        reference_boxes = _to_numpy(reference_result.boxes.xyxy)
        reference_masks = _to_numpy(reference_result.masks.data)
        selected_reference = select_mat_athlete(
            reference_boxes,
            reference_masks,
            mat_polygon,
            frame_width,
            frame_height,
        )
        reference_box = reference_boxes[selected_reference].copy()

        detections = self._track_all_people(video.path, frame_count)
        target_boxes, target_ids, observed = self._build_trajectory(
            detections,
            reference_frame_index,
            reference_box,
            mat_polygon,
            frame_width,
            frame_height,
        )
        interpolated_boxes = interpolate_box_gaps(
            target_boxes,
            self.config.maximum_tracking_gap,
        )
        output_directory = player_angle_output_directory(
            output_root,
            video.player_id,
            video.angle,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        rows = self._isolate_video(
            video.path,
            output_directory,
            segmentation_model,
            interpolated_boxes,
            target_ids,
            observed,
            mat_polygon,
            frame_width,
            frame_height,
            fps,
        )
        frames_isolated = sum(row[-1] != "missing" for row in rows)
        frames_recovered = sum(row[-1] == "recovered" for row in rows)
        result = IsolationResult(
            player_id=video.player_id,
            angle=video.angle,
            input_path=str(video.path),
            output_directory=str(output_directory),
            source_width=frame_width,
            source_height=frame_height,
            fps=fps,
            frames_processed=len(rows),
            frames_isolated=frames_isolated,
            frames_recovered=frames_recovered,
            reference_frame=reference_frame_index,
            mat_detection=mat_detection,
            device=self.device,
        )
        (output_directory / "metadata.json").write_text(
            json.dumps(asdict(result), indent=2),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _read_frame(path: Path, frame_index: int) -> np.ndarray:
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
        finally:
            capture.release()
        if not success:
            raise ValueError(f"Could not read frame {frame_index} from {path}")
        return frame

    def _track_all_people(
        self,
        video_path: Path,
        frame_count: int,
    ) -> list[dict[str, np.ndarray]]:
        tracking_model = self.model_factory(self.config.model_path)
        capture = cv2.VideoCapture(str(video_path))
        detections: list[dict[str, np.ndarray]] = []
        progress = tqdm(total=frame_count, desc=f"Tracking {video_path.name}", unit="frame")
        try:
            for _ in range(frame_count):
                success, frame = capture.read()
                if not success:
                    break
                result = tracking_model.track(
                    frame,
                    device=self.device,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=self.config.recovery_confidence,
                    verbose=False,
                )[0]
                if result.boxes is None or len(result.boxes) == 0:
                    detections.append(_empty_detections())
                else:
                    boxes = _to_numpy(result.boxes.xyxy)
                    track_ids = (
                        _to_numpy(result.boxes.id).astype(int)
                        if result.boxes.id is not None
                        else np.full(len(boxes), -1, dtype=int)
                    )
                    detections.append({"boxes": boxes, "ids": track_ids})
                progress.update(1)
        finally:
            capture.release()
            progress.close()
        return detections

    def _build_trajectory(
        self,
        detections: list[dict[str, np.ndarray]],
        reference_frame_index: int,
        reference_box: np.ndarray,
        mat_polygon: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[np.ndarray | None], list[int | None], list[bool]]:
        if reference_frame_index >= len(detections):
            raise RuntimeError("Reference frame was not reached during tracking.")
        anchor_detections = detections[reference_frame_index]
        if len(anchor_detections["boxes"]) == 0:
            raise RuntimeError("No tracked people were found in the reference frame.")
        anchor_index = select_tracking_candidate(
            anchor_detections["boxes"],
            reference_box,
            frame_width,
            frame_height,
        )
        anchor_box = anchor_detections["boxes"][anchor_index].copy()
        anchor_raw_id = int(anchor_detections["ids"][anchor_index])
        anchor_id = anchor_raw_id if anchor_raw_id >= 0 else None
        target_boxes: list[np.ndarray | None] = [None] * len(detections)
        target_ids: list[int | None] = [None] * len(detections)
        target_boxes[reference_frame_index] = anchor_box
        target_ids[reference_frame_index] = anchor_id

        def point_on_mat(point: np.ndarray) -> bool:
            return (
                cv2.pointPolygonTest(
                    mat_polygon.astype(np.float32),
                    tuple(np.asarray(point, dtype=float)),
                    False,
                )
                >= 0
            )

        def box_on_mat(box: np.ndarray) -> bool:
            return any(point_on_mat(point) for point in box_bottom_contact_points(box))

        def follow(frame_indices: Iterable[int]) -> None:
            previous_box = anchor_box.copy()
            current_id = anchor_id
            missed_frames = 0
            for frame_index in frame_indices:
                frame_detections = detections[frame_index]
                boxes = frame_detections["boxes"]
                ids = frame_detections["ids"]
                if len(boxes) == 0:
                    missed_frames += 1
                    continue
                maximum_step = 0.045 * min(missed_frames + 1, 4)
                selected: int | None = None
                if current_id is not None:
                    matches = np.flatnonzero(ids == current_id)
                    if matches.size:
                        candidate = int(matches[0])
                        if box_on_mat(boxes[candidate]) and plausible_same_person(
                            boxes[candidate],
                            previous_box,
                            frame_width,
                            frame_height,
                            maximum_center_step=maximum_step,
                        ):
                            selected = candidate
                if selected is None:
                    candidate = select_tracking_candidate(
                        boxes,
                        previous_box,
                        frame_width,
                        frame_height,
                    )
                    if not box_on_mat(boxes[candidate]) or not plausible_same_person(
                        boxes[candidate],
                        previous_box,
                        frame_width,
                        frame_height,
                        maximum_center_step=maximum_step,
                    ):
                        missed_frames += 1
                        continue
                    selected = candidate
                    if ids[selected] >= 0:
                        current_id = int(ids[selected])
                previous_box = boxes[selected].copy()
                missed_frames = 0
                target_boxes[frame_index] = previous_box
                target_ids[frame_index] = (
                    int(ids[selected]) if ids[selected] >= 0 else current_id
                )

        follow(range(reference_frame_index + 1, len(detections)))
        follow(range(reference_frame_index - 1, -1, -1))
        return target_boxes, target_ids, [box is not None for box in target_boxes]

    def _isolate_video(
        self,
        video_path: Path,
        output_directory: Path,
        model: Any,
        expected_boxes: list[np.ndarray | None],
        target_ids: list[int | None],
        observed: list[bool],
        mat_polygon: np.ndarray,
        frame_width: int,
        frame_height: int,
        fps: float,
    ) -> list[list[object]]:
        codec = cv2.VideoWriter_fourcc(*self.config.codec)
        isolated_path = output_directory / "isolated.mp4"
        crop_path = output_directory / "crop.mp4"
        tracking_path = output_directory / "tracking.csv"
        isolated_writer = cv2.VideoWriter(
            str(isolated_path),
            codec,
            fps,
            (frame_width, frame_height),
        )
        crop_writer = cv2.VideoWriter(
            str(crop_path),
            codec,
            fps,
            (self.config.output_size, self.config.output_size),
        )
        if not isolated_writer.isOpened() or not crop_writer.isOpened():
            isolated_writer.release()
            crop_writer.release()
            raise OSError(f"Could not create output videos under {output_directory}")

        def point_on_mat(point: np.ndarray) -> bool:
            return (
                cv2.pointPolygonTest(
                    mat_polygon.astype(np.float32),
                    tuple(np.asarray(point, dtype=float)),
                    False,
                )
                >= 0
            )

        def find_segmentation(
            frame: np.ndarray,
            expected_box: np.ndarray,
        ) -> tuple[Any | None, int | None]:
            attempts = (
                (self.config.output_size, self.config.confidence),
                (self.config.recovery_image_size, self.config.recovery_confidence),
            )
            for image_size, confidence in attempts:
                result = model.predict(
                    frame,
                    device=self.device,
                    classes=[0],
                    conf=confidence,
                    imgsz=image_size,
                    retina_masks=True,
                    verbose=False,
                )[0]
                if (
                    result.boxes is None
                    or len(result.boxes) == 0
                    or result.masks is None
                ):
                    continue
                boxes = _to_numpy(result.boxes.xyxy)
                masks = _to_numpy(result.masks.data)
                candidate = select_tracking_candidate(
                    boxes,
                    expected_box,
                    frame_width,
                    frame_height,
                )
                contact = mask_ground_contact(
                    masks[candidate],
                    boxes[candidate],
                    frame_width,
                    frame_height,
                )
                if point_on_mat(contact) and plausible_same_person(
                    boxes[candidate],
                    expected_box,
                    frame_width,
                    frame_height,
                    maximum_center_step=0.045,
                ):
                    return result, candidate
            return None, None

        capture = cv2.VideoCapture(str(video_path))
        rows: list[list[object]] = []
        progress = tqdm(
            total=len(expected_boxes),
            desc=f"Isolating {video_path.name}",
            unit="frame",
        )
        try:
            for frame_index, expected_box in enumerate(expected_boxes):
                success, frame = capture.read()
                if not success:
                    break
                result, selected = (
                    find_segmentation(frame, expected_box)
                    if expected_box is not None
                    else (None, None)
                )
                if result is None or selected is None:
                    isolated_writer.write(np.zeros_like(frame))
                    crop_writer.write(
                        np.zeros(
                            (self.config.output_size, self.config.output_size, 3),
                            dtype=np.uint8,
                        ),
                    )
                    rows.append(
                        [
                            frame_index,
                            (
                                target_ids[frame_index]
                                if target_ids[frame_index] is not None
                                else ""
                            ),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "missing",
                        ],
                    )
                    progress.update(1)
                    continue

                boxes = _to_numpy(result.boxes.xyxy)
                confidences = _to_numpy(result.boxes.conf)
                masks = _to_numpy(result.masks.data)
                selected_box = boxes[selected]
                foreground = isolate_person(frame, masks[selected])
                crop = padded_crop(foreground, selected_box, self.config.crop_padding)
                normalized_crop = letterbox_frame(
                    crop,
                    self.config.output_size,
                    self.config.output_size,
                )
                isolated_writer.write(foreground)
                crop_writer.write(normalized_crop)
                rows.append(
                    [
                        frame_index,
                        (
                            target_ids[frame_index]
                            if target_ids[frame_index] is not None
                            else ""
                        ),
                        float(confidences[selected]),
                        *selected_box.tolist(),
                        "tracked" if observed[frame_index] else "recovered",
                    ],
                )
                progress.update(1)
        finally:
            capture.release()
            isolated_writer.release()
            crop_writer.release()
            progress.close()

        with tracking_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["frame", "track_id", "confidence", "x1", "y1", "x2", "y2", "status"],
            )
            writer.writerows(rows)
        return rows

    @staticmethod
    def _write_player_manifest(
        output_root: Path,
        player_id: str,
        results: list[IsolationResult],
    ) -> None:
        player_directory = output_root.expanduser().resolve() / player_id
        player_directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "player_id": player_id,
            "angles": {
                result.angle: {
                    "input_path": result.input_path,
                    "output_directory": result.output_directory,
                    "frames_processed": result.frames_processed,
                    "frames_isolated": result.frames_isolated,
                }
                for result in results
            },
        }
        (player_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_dataset_manifest(
        output_root: Path,
        grouped: dict[str, list[PlayerVideo]],
        results: list[IsolationResult],
    ) -> None:
        root = output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "players": {
                player_id: [video.angle for video in videos]
                for player_id, videos in grouped.items()
            },
            "videos_processed": len(results),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
