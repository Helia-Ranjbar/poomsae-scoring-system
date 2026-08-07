from __future__ import annotations

import math

import cv2
import numpy as np


def box_iou(first_box: np.ndarray, second_box: np.ndarray) -> float:
    first = np.asarray(first_box, dtype=float)
    second = np.asarray(second_box, dtype=float)
    intersection_left = max(first[0], second[0])
    intersection_top = max(first[1], second[1])
    intersection_right = min(first[2], second[2])
    intersection_bottom = min(first[3], second[3])
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def front_candidate_scores(
    boxes: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    candidates = np.asarray(boxes, dtype=float)
    if candidates.size == 0:
        return np.empty(0, dtype=float)

    centers_x = (candidates[:, 0] + candidates[:, 2]) / 2
    bottoms = candidates[:, 3] / frame_height
    areas = (
        (candidates[:, 2] - candidates[:, 0])
        * (candidates[:, 3] - candidates[:, 1])
        / (frame_width * frame_height)
    )
    center_proximity = 1 - np.minimum(
        np.abs(centers_x - frame_width / 2) / (frame_width / 2),
        1,
    )
    normalized_area = np.sqrt(np.clip(areas, 0, 1))
    return 0.45 * bottoms + 0.35 * normalized_area + 0.20 * center_proximity


def select_front_candidate(
    boxes: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> int:
    scores = front_candidate_scores(boxes, frame_width, frame_height)
    if scores.size == 0:
        raise ValueError("No person candidates were provided.")
    return int(np.argmax(scores))


def select_tracking_candidate(
    boxes: np.ndarray,
    previous_box: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> int:
    candidates = np.asarray(boxes, dtype=float)
    if candidates.size == 0:
        raise ValueError("No person candidates were provided.")

    previous = np.asarray(previous_box, dtype=float)
    previous_center = np.array(
        [(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2],
    )
    previous_area = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
    frame_diagonal = math.hypot(frame_width, frame_height)

    scores = []
    for candidate in candidates:
        candidate_center = np.array(
            [(candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2],
        )
        center_distance = np.linalg.norm(candidate_center - previous_center) / frame_diagonal
        candidate_area = max(
            1.0,
            (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
        )
        area_similarity = math.exp(-abs(math.log(candidate_area / previous_area)))
        score = (
            0.55 * box_iou(candidate, previous)
            + 0.30 * max(0.0, 1 - center_distance * 4)
            + 0.15 * area_similarity
        )
        scores.append(score)
    return int(np.argmax(scores))


def box_bottom_contact_points(
    box: np.ndarray,
    inset_ratio: float = 0.15,
) -> np.ndarray:
    x1, _, x2, y2 = np.asarray(box, dtype=float)
    inset = max(0.0, min(0.5, inset_ratio)) * max(0.0, x2 - x1)
    return np.array(
        [
            [x1 + inset, y2],
            [(x1 + x2) / 2, y2],
            [x2 - inset, y2],
        ],
        dtype=float,
    )


def interpolate_box_gaps(
    boxes: list[np.ndarray | None],
    max_gap: int,
) -> list[np.ndarray | None]:
    interpolated = [
        None if box is None else np.asarray(box, dtype=float).copy()
        for box in boxes
    ]
    valid_indices = [
        index for index, box in enumerate(interpolated) if box is not None
    ]
    for left_index, right_index in zip(valid_indices, valid_indices[1:]):
        gap = right_index - left_index - 1
        if gap <= 0 or gap > max_gap:
            continue
        left_box = interpolated[left_index]
        right_box = interpolated[right_index]
        for offset in range(1, gap + 1):
            weight = offset / (gap + 1)
            interpolated[left_index + offset] = (
                (1 - weight) * left_box + weight * right_box
            )
    return interpolated


def isolate_person(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary_mask = np.asarray(mask) > 0.5
    if binary_mask.shape != frame.shape[:2]:
        binary_mask = cv2.resize(
            binary_mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    isolated = np.zeros_like(frame)
    isolated[binary_mask] = frame[binary_mask]
    return isolated


def padded_crop(
    frame: np.ndarray,
    box: np.ndarray,
    padding_ratio: float = 0.20,
) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    width = x2 - x1
    height = y2 - y1
    x_padding = width * padding_ratio
    y_padding = height * padding_ratio
    frame_height, frame_width = frame.shape[:2]
    left = max(0, round(x1 - x_padding))
    top = max(0, round(y1 - y_padding))
    right = min(frame_width, round(x2 + x_padding))
    bottom = min(frame_height, round(y2 + y_padding))
    return frame[top:bottom, left:right]
