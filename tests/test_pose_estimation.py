import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poomsae_scoring.scoring.pose_estimation import (
    MEDIAPIPE_LANDMARK_INDEX,
    PoseConfig,
    discover_pose_videos,
    mediapipe_landmarks_to_arrays,
    mediapipe_to_coco,
    pose_output_is_complete,
)


def test_discover_pose_videos_finds_player_angles_in_natural_order(tmp_path: Path) -> None:
    for relative_path in (
        "WP10/angles/45/crop.mp4",
        "WP2/angles/135/crop.mp4",
        "WP2/angles/0/crop.mp4",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True)
        path.touch()

    videos = discover_pose_videos(tmp_path)

    assert [(video.player_id, video.angle) for video in videos] == [
        ("WP2", "0"),
        ("WP2", "135"),
        ("WP10", "45"),
    ]


def test_discover_pose_videos_can_filter_angles(tmp_path: Path) -> None:
    for angle in ("0", "45", "135"):
        path = tmp_path / "WP1" / "angles" / angle / "crop.mp4"
        path.parent.mkdir(parents=True)
        path.touch()

    videos = discover_pose_videos(tmp_path, angles={"45"})

    assert [video.angle for video in videos] == ["45"]


def test_mediapipe_landmarks_convert_to_pixel_coordinates() -> None:
    landmarks = [
        SimpleNamespace(x=0.25, y=0.5, z=-0.1, visibility=0.9)
        for _ in range(33)
    ]

    keypoints, confidence, z_coordinates = mediapipe_landmarks_to_arrays(
        landmarks,
        width=640,
        height=480,
    )

    np.testing.assert_allclose(keypoints[0], [160, 240])
    assert confidence[0] == pytest.approx(0.9)
    assert z_coordinates[0] == pytest.approx(-64)


def test_mediapipe_landmarks_map_to_coco_for_existing_features() -> None:
    keypoints = np.zeros((33, 2), dtype=float)
    confidence = np.arange(33, dtype=float) / 32
    left_shoulder = MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]
    keypoints[left_shoulder] = [123, 234]

    coco_keypoints, coco_confidence = mediapipe_to_coco(keypoints, confidence)

    np.testing.assert_allclose(coco_keypoints[5], [123, 234])
    assert coco_confidence[5] == confidence[left_shoulder]


def test_pose_output_complete_requires_all_files(tmp_path: Path) -> None:
    crop = tmp_path / "WP1" / "angles" / "45" / "crop.mp4"
    crop.parent.mkdir(parents=True)
    crop.touch()
    video = discover_pose_videos(tmp_path)[0]
    video.output_directory.mkdir()
    for filename in ("keypoints.csv", "features.csv"):
        (video.output_directory / filename).touch()
    with (video.output_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"completed_full_video": True, "backend": "MediaPipe Pose"},
            file,
        )

    assert not pose_output_is_complete(video)
    assert pose_output_is_complete(video, save_annotated_video=False)


def test_pose_output_complete_rejects_partial_smoke_test(tmp_path: Path) -> None:
    crop = tmp_path / "WP1" / "angles" / "45" / "crop.mp4"
    crop.parent.mkdir(parents=True)
    crop.touch()
    video = discover_pose_videos(tmp_path)[0]
    video.output_directory.mkdir()
    for filename in ("keypoints.csv", "features.csv"):
        (video.output_directory / filename).touch()
    with (video.output_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"completed_full_video": False}, file)

    assert not pose_output_is_complete(video, save_annotated_video=False)


def test_pose_config_rejects_invalid_model_complexity() -> None:
    with pytest.raises(ValueError, match="complexity"):
        PoseConfig(model_complexity=3)
