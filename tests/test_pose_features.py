import numpy as np

from poomsae_scoring.scoring.pose_features import (
    extract_pose_features,
    find_low_motion_intervals,
    pose_motion_energy,
)


def sample_pose() -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[0] = [50, 10]
    keypoints[5] = [40, 30]
    keypoints[6] = [60, 30]
    keypoints[7] = [35, 50]
    keypoints[8] = [65, 50]
    keypoints[9] = [30, 70]
    keypoints[10] = [70, 70]
    keypoints[11] = [42, 70]
    keypoints[12] = [58, 70]
    keypoints[13] = [42, 100]
    keypoints[14] = [58, 100]
    keypoints[15] = [42, 130]
    keypoints[16] = [58, 130]
    confidence = np.ones(17, dtype=float)
    return keypoints, confidence


def test_extract_pose_features_finds_straight_knees_and_centered_torso() -> None:
    keypoints, confidence = sample_pose()

    features = extract_pose_features(keypoints, confidence)

    assert features["left_knee_angle"] == 180.0
    assert features["right_knee_angle"] == 180.0
    assert features["torso_lean_degrees"] == 0.0
    assert features["support_side"] == "left"


def test_motion_energy_increases_with_keypoint_displacement() -> None:
    previous, confidence = sample_pose()
    current = previous.copy()
    current[:, 0] += 2

    energy = pose_motion_energy(previous, current, confidence, confidence, fps=30)

    assert energy > 0


def test_find_low_motion_intervals_reports_three_second_pause() -> None:
    motion = [0.2] * 30 + [0.01] * 90 + [0.2] * 30

    intervals = find_low_motion_intervals(motion, fps=30)

    assert intervals == [(1.0, 4.0)]

