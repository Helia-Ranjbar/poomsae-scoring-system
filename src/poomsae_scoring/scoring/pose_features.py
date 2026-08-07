from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KEYPOINT_INDEX = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}


def _point(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    name: str,
    minimum_confidence: float,
) -> np.ndarray | None:
    index = KEYPOINT_INDEX[name]
    if confidence[index] < minimum_confidence:
        return None
    return keypoints[index].astype(float)


def _distance(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return math.nan
    return float(np.linalg.norm(first - second))


def _midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return (first + second) / 2


def _angle(
    first: np.ndarray | None,
    vertex: np.ndarray | None,
    third: np.ndarray | None,
) -> float:
    if first is None or vertex is None or third is None:
        return math.nan
    first_vector = first - vertex
    second_vector = third - vertex
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0:
        return math.nan
    cosine = np.clip(np.dot(first_vector, second_vector) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _line_angle(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return math.nan
    delta = second - first
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def extract_pose_features(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    minimum_confidence: float = 0.25,
) -> dict[str, float | str]:
    points = np.asarray(keypoints, dtype=float)
    scores = np.asarray(confidence, dtype=float)
    if points.shape != (17, 2):
        raise ValueError("Expected COCO pose keypoints with shape (17, 2).")
    if scores.shape != (17,):
        raise ValueError("Expected keypoint confidence with shape (17,).")

    detected = {
        name: _point(points, scores, name, minimum_confidence)
        for name in COCO_KEYPOINT_NAMES
    }
    shoulder_midpoint = _midpoint(
        detected["left_shoulder"],
        detected["right_shoulder"],
    )
    hip_midpoint = _midpoint(detected["left_hip"], detected["right_hip"])
    shoulder_width = _distance(
        detected["left_shoulder"],
        detected["right_shoulder"],
    )
    body_scale = shoulder_width if math.isfinite(shoulder_width) and shoulder_width > 0 else 1.0

    left_knee_angle = _angle(
        detected["left_hip"],
        detected["left_knee"],
        detected["left_ankle"],
    )
    right_knee_angle = _angle(
        detected["right_hip"],
        detected["right_knee"],
        detected["right_ankle"],
    )
    left_ankle = detected["left_ankle"]
    right_ankle = detected["right_ankle"]
    support_side = "unknown"
    if left_ankle is not None and right_ankle is not None:
        support_side = "left" if left_ankle[1] >= right_ankle[1] else "right"

    support_knee_angle = (
        left_knee_angle
        if support_side == "left"
        else right_knee_angle if support_side == "right" else math.nan
    )
    kicking_knee_angle = (
        right_knee_angle
        if support_side == "left"
        else left_knee_angle if support_side == "right" else math.nan
    )

    torso_lean = math.nan
    if shoulder_midpoint is not None and hip_midpoint is not None:
        torso_vector = shoulder_midpoint - hip_midpoint
        torso_lean = float(
            abs(np.degrees(np.arctan2(torso_vector[0], -torso_vector[1]))),
        )

    stance_width = _distance(left_ankle, right_ankle) / body_scale
    body_center_x = hip_midpoint[0] if hip_midpoint is not None else math.nan
    support_ankle = (
        left_ankle
        if support_side == "left"
        else right_ankle if support_side == "right" else None
    )
    balance_offset = (
        abs(body_center_x - support_ankle[0]) / body_scale
        if support_ankle is not None and math.isfinite(body_center_x)
        else math.nan
    )

    def centerline_offset(point: np.ndarray | None) -> float:
        if point is None or not math.isfinite(body_center_x):
            return math.nan
        return float((point[0] - body_center_x) / body_scale)

    return {
        "left_elbow_angle": _angle(
            detected["left_shoulder"],
            detected["left_elbow"],
            detected["left_wrist"],
        ),
        "right_elbow_angle": _angle(
            detected["right_shoulder"],
            detected["right_elbow"],
            detected["right_wrist"],
        ),
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "support_knee_angle": support_knee_angle,
        "kicking_knee_angle": kicking_knee_angle,
        "support_side": support_side,
        "torso_lean_degrees": torso_lean,
        "shoulder_tilt_degrees": _line_angle(
            detected["left_shoulder"],
            detected["right_shoulder"],
        ),
        "hip_tilt_degrees": _line_angle(
            detected["left_hip"],
            detected["right_hip"],
        ),
        "stance_width_shoulders": stance_width,
        "balance_offset_shoulders": balance_offset,
        "left_wrist_centerline": centerline_offset(detected["left_wrist"]),
        "right_wrist_centerline": centerline_offset(detected["right_wrist"]),
        "left_ankle_centerline": centerline_offset(left_ankle),
        "right_ankle_centerline": centerline_offset(right_ankle),
        "head_centerline": centerline_offset(detected["nose"]),
        "mean_keypoint_confidence": float(np.mean(scores)),
    }


def pose_motion_energy(
    previous_keypoints: np.ndarray,
    current_keypoints: np.ndarray,
    previous_confidence: np.ndarray,
    current_confidence: np.ndarray,
    fps: float,
    minimum_confidence: float = 0.25,
) -> float:
    previous = np.asarray(previous_keypoints, dtype=float)
    current = np.asarray(current_keypoints, dtype=float)
    previous_scores = np.asarray(previous_confidence, dtype=float)
    current_scores = np.asarray(current_confidence, dtype=float)
    valid = (previous_scores >= minimum_confidence) & (current_scores >= minimum_confidence)
    if not np.any(valid):
        return math.nan

    left_shoulder = current[KEYPOINT_INDEX["left_shoulder"]]
    right_shoulder = current[KEYPOINT_INDEX["right_shoulder"]]
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale <= 0:
        return math.nan
    displacement = np.linalg.norm(current[valid] - previous[valid], axis=1) / scale
    return float(np.mean(displacement) * fps)


def find_low_motion_intervals(
    motion_energy: Sequence[float],
    fps: float,
    threshold: float = 0.08,
    minimum_seconds: float = 3.0,
) -> list[tuple[float, float]]:
    values = np.asarray(motion_energy, dtype=float)
    minimum_frames = max(1, round(minimum_seconds * fps))
    intervals: list[tuple[float, float]] = []
    run_start: int | None = None

    for index, value in enumerate(values):
        is_low_motion = math.isfinite(value) and value < threshold
        if is_low_motion and run_start is None:
            run_start = index
        if not is_low_motion and run_start is not None:
            if index - run_start >= minimum_frames:
                intervals.append((run_start / fps, index / fps))
            run_start = None

    if run_start is not None and len(values) - run_start >= minimum_frames:
        intervals.append((run_start / fps, len(values) / fps))
    return intervals

